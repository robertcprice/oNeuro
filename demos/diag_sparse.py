#!/usr/bin/env python3
"""Verify the sparse matrix contains correct values after weight programming."""

import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from oneuro.molecular.cuda_backend import CUDARegionalBrain, NT_GABA

device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
rb = CUDARegionalBrain.medium(device=device, seed=42)
brain = rb.brain
dev = brain.device

relay_ids = torch.tensor(rb.regions["thalamus"]["subgroups"]["relay"], dtype=torch.int64, device=dev)
l5_ids = []
for name, r in rb.regions.items():
    if r["type"] == "cortex" and "L5" in r["subgroups"]:
        l5_ids.extend(r["subgroups"]["L5"])
output_ids = torch.tensor(l5_ids, dtype=torch.int64, device=dev)

n_relay, n_out = len(relay_ids), len(output_ids)
words = ["cat", "dog"]
n_per_in, n_per_out = n_relay // 8, n_out // 8

# Build sparse W first (before modifications)
brain._build_sparse_W()
W = brain._W_sparse

# Verify W is populated
print(f"W shape: {W.shape}, nnz: {W._nnz()}")
print(f"Total synapses: {brain.n_synapses}")

# Simulate "cat" firing: relay neurons 0-17
cat_fired = torch.zeros(brain.n, device=dev)
for j in range(n_per_in):
    cat_fired[relay_ids[j]] = 1.0

# W @ fired = current delivered to each neuron
result = torch.sparse.mm(W, cat_fired.unsqueeze(1)).squeeze(1)
print(f"\nBefore weight modification:")
print(f"  Cat relay fires → current to cat's L5 (first 125): {result[output_ids[:n_per_out]].sum():.1f}")
print(f"  Cat relay fires → current to dog's L5 (next 125): {result[output_ids[n_per_out:2*n_per_out]].sum():.1f}")
print(f"  Cat relay fires → current to ALL L5: {result[output_ids].sum():.1f}")

# Now program weights: matching = 8.0, non-matching = 0.3
relay_set = set(relay_ids.cpu().tolist())
l5_set = set(output_ids.cpu().tolist())

pre_is_relay = torch.zeros(brain.n_synapses, dtype=torch.bool, device=dev)
post_is_l5 = torch.zeros(brain.n_synapses, dtype=torch.bool, device=dev)
for rid in relay_set:
    pre_is_relay |= (brain.syn_pre == rid)
for lid in l5_set:
    post_is_l5 |= (brain.syn_post == lid)
relay_l5_mask = pre_is_relay & post_is_l5 & (brain.syn_nt_type != NT_GABA)

# Set all relay→L5 to 0.3
brain.syn_strength[relay_l5_mask] = 0.3

# Cat matching: relay 0-17 → L5 0-124
for i, w in enumerate(words):
    in_start = i * n_per_in
    in_end = min(in_start + n_per_in, n_relay)
    out_start = i * n_per_out
    out_end = min(out_start + n_per_out, n_out)

    word_relay_set = set(relay_ids[in_start:in_end].cpu().tolist())
    word_l5_set = set(output_ids[out_start:out_end].cpu().tolist())

    pre_match = torch.zeros(brain.n_synapses, dtype=torch.bool, device=dev)
    post_match = torch.zeros(brain.n_synapses, dtype=torch.bool, device=dev)
    for rid in word_relay_set:
        pre_match |= (brain.syn_pre == rid)
    for lid in word_l5_set:
        post_match |= (brain.syn_post == lid)

    match = pre_match & post_match & (brain.syn_nt_type != NT_GABA)
    brain.syn_strength[match] = 8.0
    print(f"\n{w}: {int(match.sum())} synapses set to 8.0")

# Verify strength values
cat_relay_mask = torch.zeros(brain.n_synapses, dtype=torch.bool, device=dev)
for rid in relay_ids[:n_per_in].cpu().tolist():
    cat_relay_mask |= (brain.syn_pre == rid)

cat_l5_mask = torch.zeros(brain.n_synapses, dtype=torch.bool, device=dev)
for lid in output_ids[:n_per_out].cpu().tolist():
    cat_l5_mask |= (brain.syn_post == lid)

matching = cat_relay_mask & cat_l5_mask & (brain.syn_nt_type != NT_GABA)
nonmatching = cat_relay_mask & post_is_l5 & ~cat_l5_mask & (brain.syn_nt_type != NT_GABA)

print(f"\nDirect synapse check:")
print(f"  Cat relay → cat L5 synapses: {int(matching.sum())}, strength: {brain.syn_strength[matching].mean():.1f}")
print(f"  Cat relay → other L5 synapses: {int(nonmatching.sum())}, strength: {brain.syn_strength[nonmatching].mean():.1f}")

# Rebuild sparse W
brain._W_dirty = True
brain._W_sparse = None
brain._NT_W_sparse = None
brain._build_sparse_W()
W = brain._W_sparse

# Re-test with new W
result = torch.sparse.mm(W, cat_fired.unsqueeze(1)).squeeze(1)
print(f"\nAfter weight modification + sparse rebuild:")
print(f"  Cat relay fires → current to cat's L5 (first {n_per_out}): {result[output_ids[:n_per_out]].sum():.1f}")
print(f"  Cat relay fires → current to dog's L5 (next {n_per_out}): {result[output_ids[n_per_out:2*n_per_out]].sum():.1f}")
print(f"  Cat relay fires → current to ALL L5: {result[output_ids].sum():.1f}")
ratio = float(result[output_ids[:n_per_out]].sum() / max(result[output_ids[n_per_out:2*n_per_out]].sum(), 0.01))
print(f"  Ratio (cat L5 / dog L5): {ratio:.2f}x")

# Check total current at a single cat L5 neuron vs dog L5 neuron
cat_l5_0 = output_ids[0]
dog_l5_0 = output_ids[n_per_out]
print(f"\n  Single neuron current: cat_L5[0]={result[cat_l5_0]:.2f}, dog_L5[0]={result[dog_l5_0]:.2f}")

# Now scale by psc_scale to get actual uA
psc = brain.psc_scale
print(f"  Actual current (×psc_scale={psc}): cat_L5[0]={result[cat_l5_0]*psc:.1f}µA, dog_L5[0]={result[dog_l5_0]*psc:.1f}µA")
