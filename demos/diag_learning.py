#!/usr/bin/env python3
"""Word learning demo with weight-based readout.

The brain runs a full biological simulation (HH dynamics, calcium, STDP, etc.).
Training modifies synapse weights via Hebbian learning + DA-gated STDP.
Word identification uses weight-based readout from the relay→L5 pathway.

This is how real BCI/BMI decoders work: the brain does the computation,
a decoder reads the neural state.

Protocol:
1. Build brain, warmup
2. Train words using Hebbian + STDP
3. Readout: sum of relay→L5 synapse strengths per word group
4. Sentence formation: temporal chains via word-pair training
"""

import os, sys, time
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from oneuro.molecular.cuda_backend import (
    CUDARegionalBrain, CUDAMolecularBrain, NT_DA, NT_GLU, NT_GABA,
)


def weight_readout(
    brain: CUDAMolecularBrain,
    word_relay_ids: torch.Tensor,
    all_target_groups: dict,  # {word: tensor of L5 neuron ids}
    device: torch.device,
) -> dict:
    """Read the word→meaning associations directly from synapse weights.

    For each word W: sum strengths of synapses from W's relay neurons → each target group.
    The group with the highest total = the decoded word.
    """
    scores = {}
    for target_word, target_l5_ids in all_target_groups.items():
        # Build masks for this target group
        target_set = torch.zeros(brain.n, dtype=torch.bool, device=device)
        target_set[target_l5_ids] = True
        post_is_target = target_set[brain.syn_post]

        # Synapses from word's relay → this target group
        pre_is_word = torch.zeros(brain.n, dtype=torch.bool, device=device)
        pre_is_word[word_relay_ids] = True
        pre_match = pre_is_word[brain.syn_pre]

        matching = pre_match & post_is_target & (brain.syn_nt_type != NT_GABA)
        if matching.any():
            total = float((brain.syn_strength[matching] * brain.syn_weight[matching]).sum())
        else:
            total = 0.0
        scores[target_word] = total

    return scores


def main():
    device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
    print(f"Device: {device}")

    print("Building medium brain (~5K neurons)...")
    t0 = time.time()
    rb = CUDARegionalBrain.medium(device=device, seed=42)
    brain = rb.brain
    dev = brain.device
    print(f"  {brain.n} neurons, {brain.n_synapses} synapses")

    relay_ids = torch.tensor(
        rb.regions["thalamus"]["subgroups"]["relay"], dtype=torch.int64, device=dev
    )
    l5_ids = []
    for name, r in rb.regions.items():
        if r["type"] == "cortex" and "L5" in r["subgroups"]:
            l5_ids.extend(r["subgroups"]["L5"])
    output_ids = torch.tensor(l5_ids, dtype=torch.int64, device=dev)
    n_relay = len(relay_ids)
    n_out = len(output_ids)

    # ══════════════════════════════════════════════════════════════
    # Vocabulary setup: 8 words, non-overlapping input + output
    # ══════════════════════════════════════════════════════════════
    words = ["cat", "dog", "bird", "fish", "run", "eat", "big", "red"]
    n_words = len(words)
    n_per_in = n_relay // n_words
    n_per_out = n_out // n_words
    print(f"  {n_relay} relay, {n_out} L5, {n_per_in} relay/word, {n_per_out} L5/word")

    rng = np.random.RandomState(42)

    # Non-overlapping input patterns
    input_patterns = {}
    word_relay_map = {}  # {word: tensor of relay neuron IDs}
    for i, w in enumerate(words):
        p = np.zeros(n_relay, dtype=np.float32)
        start = i * n_per_in
        end = min(start + n_per_in, n_relay)
        for j in range(start, end):
            p[j] = 0.8 + rng.uniform(0, 0.2)
        input_patterns[w] = torch.tensor(p, device=dev)
        word_relay_map[w] = relay_ids[start:end]

    # Non-overlapping target groups
    target_patterns = {}
    word_target_map = {}  # {word: tensor of L5 neuron IDs}
    for i, w in enumerate(words):
        p = torch.zeros(n_out, device=dev)
        start = i * n_per_out
        end = min(start + n_per_out, n_out)
        p[start:end] = 1.0
        target_patterns[w] = p
        word_target_map[w] = output_ids[start:end]

    # ══════════════════════════════════════════════════════════════
    # Phase 1: Pre-training readout
    # ══════════════════════════════════════════════════════════════
    print("\n── Phase 1: Pre-training weight readout ──")
    pre_correct = 0
    for w in words:
        scores = weight_readout(brain, word_relay_map[w], word_target_map, dev)
        best_w = max(scores, key=scores.get)
        own_score = scores[w]
        best_score = scores[best_w]
        match = "✓" if best_w == w else "✗"
        if best_w == w: pre_correct += 1
        print(f"  {w:6s} → {best_w:6s} (own={own_score:.1f}, best={best_score:.1f}) {match}")
    print(f"  Pre-training accuracy: {pre_correct}/{n_words}")

    # ══════════════════════════════════════════════════════════════
    # Phase 2: Warmup + Training
    # ══════════════════════════════════════════════════════════════
    print("\n── Phase 2: Warmup ──")
    for s in range(500):
        if s % 4 == 0:
            rb.stimulate_thalamus(15.0)
        rb.step()

    print(f"\n── Phase 3: Training (30 reps × {n_words} words) ──")
    t1 = time.time()

    for rep in range(30):
        word_order = list(words)
        rng.shuffle(word_order)

        for w in word_order:
            rb.train_word(
                relay_ids, input_patterns[w],
                output_ids, target_patterns[w],
                train_steps=80,
                input_intensity=70.0,
                target_intensity=60.0,
                da_amount=80.0,
                hebbian_delta=0.5,
            )
            rb.run(10)

        if (rep + 1) % 10 == 0:
            s_mean = float(brain.syn_strength.mean())
            s_std = float(brain.syn_strength.std())
            print(f"  Rep {rep+1}: strength mean={s_mean:.3f} std={s_std:.3f}")

    print(f"  Training: {time.time()-t1:.1f}s")

    # ══════════════════════════════════════════════════════════════
    # Phase 4: Hippocampal replay
    # ══════════════════════════════════════════════════════════════
    print("\n── Phase 4: Hippocampal replay ──")
    t2 = time.time()
    word_input_map = {w: (relay_ids, input_patterns[w]) for w in words}
    word_target_map_replay = {w: (output_ids, target_patterns[w]) for w in words}
    rb.consolidation_sleep(word_input_map, word_target_map_replay, n_replays=5, replay_steps=30)
    print(f"  Replay: {time.time()-t2:.1f}s")

    # ══════════════════════════════════════════════════════════════
    # Phase 5: Post-training readout
    # ══════════════════════════════════════════════════════════════
    print("\n── Phase 5: Post-training weight readout ──")
    post_correct = 0
    for w in words:
        scores = weight_readout(brain, word_relay_map[w], word_target_map, dev)
        best_w = max(scores, key=scores.get)
        own_score = scores[w]
        best_score = scores[best_w]
        match = "✓" if best_w == w else "✗"
        if best_w == w: post_correct += 1
        # Show own vs runner-up
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        runner_up = sorted_scores[1] if sorted_scores[0][0] == w else sorted_scores[0]
        margin = own_score - runner_up[1]
        print(f"  {w:6s} → {best_w:6s} own={own_score:.1f} vs {runner_up[0]}={runner_up[1]:.1f}"
              f" margin={margin:+.1f} {match}")
    print(f"  Post-training accuracy: {post_correct}/{n_words}")

    # ══════════════════════════════════════════════════════════════
    # Phase 6: Sentence chain formation
    # ══════════════════════════════════════════════════════════════
    print("\n── Phase 6: Sentence chain training ──")
    # Train temporal pairs: A→B
    sentences = [
        ("cat", "eat", "fish"),
        ("dog", "run", "big"),
        ("bird", "eat", "red"),
    ]

    # Train word-pair associations: A→B means A's relay should also activate B's L5.
    # We train this with more reps and stronger delta to overcome word-identity training.
    print("  Training word-pair associations (10 reps per pair)...")
    for rep in range(10):
        for sentence in sentences:
            for k in range(len(sentence) - 1):
                w_a, w_b = sentence[k], sentence[k + 1]
                rb.train_word(
                    relay_ids, input_patterns[w_a],
                    output_ids, target_patterns[w_b],
                    train_steps=60,
                    input_intensity=70.0,
                    target_intensity=60.0,
                    da_amount=80.0,
                    hebbian_delta=0.5,
                )
                rb.run(5)
    for sentence in sentences:
        print(f"  Trained: {' → '.join(sentence)}")

    # Test chain recall — exclude current word (identity always wins)
    print("\n  Testing chain recall (weight readout, excluding self):")
    chain_correct = 0
    chain_total = 0
    for sentence in sentences:
        chain = [sentence[0]]
        current_word = sentence[0]
        for _ in range(len(sentence) - 1):
            scores = weight_readout(brain, word_relay_map[current_word], word_target_map, dev)
            # Exclude current word — identity always dominates
            del scores[current_word]
            next_word = max(scores, key=scores.get)
            chain.append(next_word)
            current_word = next_word
            chain_total += 1
        expected = " → ".join(sentence)
        got = " → ".join(chain)
        n_match = sum(1 for a, b in zip(chain[1:], sentence[1:]) if a == b)
        chain_correct += n_match
        match = "✓" if chain == list(sentence) else "~" if chain[1] == sentence[1] else "✗"
        print(f"  Expected: {expected}")
        print(f"  Got:      {got}  {match}")

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Pre-training word accuracy:  {pre_correct}/{n_words} ({100*pre_correct/n_words:.0f}%)")
    print(f"  Post-training word accuracy: {post_correct}/{n_words} ({100*post_correct/n_words:.0f}%)")
    print(f"  Sentence chain accuracy:     {chain_correct}/{chain_total} ({100*chain_correct/max(chain_total,1):.0f}%)")
    improvement = post_correct - pre_correct
    if improvement > 0:
        print(f"  → IMPROVEMENT: +{improvement} words learned")
    synapse_str = float(brain.syn_strength.mean())
    print(f"  Synapse strength: {synapse_str:.3f} (from 1.000)")
    print(f"  Total time: {time.time()-t0:.1f}s")

    word_pass = post_correct >= 6
    chain_pass = chain_correct >= chain_total // 2
    if word_pass and chain_pass:
        print("\n  ★ PASS: Words + sentence chains learned!")
    elif word_pass:
        print("\n  ★ PASS: Words learned, chains need work")
    elif post_correct > pre_correct:
        print("\n  ~ PARTIAL: Some learning detected")
    else:
        print("\n  ✗ FAIL: No significant learning")


if __name__ == "__main__":
    main()
