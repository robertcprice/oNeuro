#!/usr/bin/env python3
"""Architecture test: Can direct weight programming produce word-specific L5 responses?

If YES → architecture works, learning algorithm needs fixing
If NO → architecture itself cannot support word discrimination

This bypasses all learning and directly sets synapse strengths.
"""

import os, sys, time
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from oneuro.molecular.cuda_backend import (
    CUDARegionalBrain, NT_DA, NT_GLU, NT_GABA,
)


def cosine_sim(a, b):
    a_f, b_f = a.float(), b.to(a.device).float()
    return float((a_f * b_f).sum() / (a_f.norm().clamp(1e-8) * b_f.norm().clamp(1e-8)))


def test_word(rb, relay_ids, input_pattern, output_ids, stim_steps=80, intensity=70.0):
    """Test with early-window spike counting to avoid reverberation noise."""
    brain = rb.brain
    dev = brain.device
    # Count spikes in early window (first 15 steps) and full window separately
    early_counts = torch.zeros(len(output_ids), device=dev)
    full_counts = torch.zeros(len(output_ids), device=dev)
    inp_active = (input_pattern > 0.3)[:len(relay_ids)]
    for s in range(stim_steps):
        if s % 2 == 0 and inp_active.any():
            active = relay_ids[inp_active]
            brain.external_current[active] += input_pattern[inp_active] * intensity
        rb.step()
        fired = brain.fired[output_ids].float()
        full_counts += fired
        if s < 15:
            early_counts += fired
    return early_counts, full_counts


def main():
    device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
    print(f"Device: {device}")

    rb = CUDARegionalBrain.medium(device=device, seed=42)
    brain = rb.brain
    dev = brain.device
    print(f"Brain: {brain.n} neurons, {brain.n_synapses} synapses")

    relay_ids = torch.tensor(
        rb.regions["thalamus"]["subgroups"]["relay"], dtype=torch.int64, device=dev
    )
    l5_ids = []
    for name, r in rb.regions.items():
        if r["type"] == "cortex" and "L5" in r["subgroups"]:
            l5_ids.extend(r["subgroups"]["L5"])
    output_ids = torch.tensor(l5_ids, dtype=torch.int64, device=dev)

    words = ["cat", "dog", "bird", "fish", "run", "eat", "big", "red"]
    n_words = len(words)
    n_relay = len(relay_ids)
    n_out = len(output_ids)
    n_per_in = n_relay // n_words
    n_per_out = n_out // n_words

    print(f"Relay: {n_relay}, L5: {n_out}")
    print(f"Per word: {n_per_in} relay, {n_per_out} L5")

    # Non-overlapping input patterns
    rng = np.random.RandomState(42)
    input_patterns = {}
    for i, w in enumerate(words):
        p = np.zeros(n_relay, dtype=np.float32)
        start = i * n_per_in
        end = min(start + n_per_in, n_relay)
        for j in range(start, end):
            p[j] = 0.8 + rng.uniform(0, 0.2)
        input_patterns[w] = torch.tensor(p, device=dev)

    # Non-overlapping target patterns
    target_patterns = {}
    for i, w in enumerate(words):
        p = torch.zeros(n_out, device=dev)
        start = i * n_per_out
        end = min(start + n_per_out, n_out)
        p[start:end] = 1.0
        target_patterns[w] = p

    # ── Direct weight programming (discriminative) ──
    print("\n── Programming weights directly (discriminative) ──")

    # First: identify all relay→L5 synapses
    relay_set = set(relay_ids.cpu().tolist())
    l5_set = set(output_ids.cpu().tolist())

    pre_is_relay = torch.zeros(brain.n_synapses, dtype=torch.bool, device=dev)
    post_is_l5 = torch.zeros(brain.n_synapses, dtype=torch.bool, device=dev)
    for rid in relay_set:
        pre_is_relay |= (brain.syn_pre == rid)
    for lid in l5_set:
        post_is_l5 |= (brain.syn_post == lid)

    all_relay_l5 = pre_is_relay & post_is_l5 & (brain.syn_nt_type != NT_GABA)
    print(f"  Total relay→L5 excitatory synapses: {int(all_relay_l5.sum())}")

    # Set ALL relay→L5 to minimum first (discriminative baseline)
    brain.syn_strength[all_relay_l5] = 0.3

    # Then for each word: strengthen matching relay→L5 to 8.0
    for i, w in enumerate(words):
        in_start = i * n_per_in
        in_end = min(in_start + n_per_in, n_relay)
        out_start = i * n_per_out
        out_end = min(out_start + n_per_out, n_out)

        word_relay_ids = set(relay_ids[in_start:in_end].cpu().tolist())
        word_l5_ids = set(output_ids[out_start:out_end].cpu().tolist())

        pre_is_word = torch.zeros(brain.n_synapses, dtype=torch.bool, device=dev)
        post_is_word = torch.zeros(brain.n_synapses, dtype=torch.bool, device=dev)
        for rid in word_relay_ids:
            pre_is_word |= (brain.syn_pre == rid)
        for lid in word_l5_ids:
            post_is_word |= (brain.syn_post == lid)

        match = pre_is_word & post_is_word & (brain.syn_nt_type != NT_GABA)
        n_match = int(match.sum())
        brain.syn_strength[match] = 8.0
        print(f"  {w}: {n_match} matching synapses → 8.0")

    # Force rebuild of sparse matrices with new weights
    brain._W_dirty = True
    brain._W_sparse = None
    brain._NT_W_sparse = None
    brain._build_sparse_W()
    print(f"  Sparse matrices rebuilt")

    # ── Warmup ──
    for s in range(300):
        if s % 4 == 0:
            rb.stimulate_thalamus(15.0)
        rb.step()

    # ── Test ──
    print("\n── Testing programmed network ──")
    print("\n  EARLY window (first 15 steps — before reverberation):")
    early_correct = 0
    for w in words:
        early_resp, full_resp = test_word(rb, relay_ids, input_patterns[w], output_ids)
        rb.run(30)

        # Early window matching
        best_w, best_sim = "", -1.0
        for cw, tp in target_patterns.items():
            sim = cosine_sim(early_resp, tp)
            if sim > best_sim:
                best_sim = sim
                best_w = cw
        own_sim = cosine_sim(early_resp, target_patterns[w])
        spikes = int(early_resp.sum())
        match = "✓" if best_w == w else "✗"
        if best_w == w: early_correct += 1

        i = words.index(w)
        s_start = i * n_per_out
        s_end = min(s_start + n_per_out, n_out)
        own_spikes = float(early_resp[s_start:s_end].sum())
        pct = 100 * own_spikes / max(spikes, 1)

        # Also show full window
        full_spikes = int(full_resp.sum())
        full_own = float(full_resp[s_start:s_end].sum())
        full_pct = 100 * full_own / max(full_spikes, 1)

        print(f"  {w:6s} → {best_w:6s} sim={best_sim:.3f} early_spikes={spikes}"
              f" early_own={pct:.1f}% | full_own={full_pct:.1f}% {match}")

    print(f"\n  Early-window accuracy: {early_correct}/{n_words} ({100*early_correct/n_words:.0f}%)")

    if correct >= 4:
        print("  ★ Architecture WORKS — learning rule needs fixing")
    else:
        print("  ✗ Architecture CANNOT support word discrimination")
        print("    Need to change architecture (more direct connections, etc.)")


if __name__ == "__main__":
    main()
