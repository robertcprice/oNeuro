#!/usr/bin/env python3
"""Benchmark: MolecularNeuralNetwork vs OrganicNeuralNetwork.

Compares both network types on XOR and PatternRecognition tasks using
identical geometry, seed, and training protocol.

Usage:
    cd oNeuro && PYTHONPATH=src python3 experiments/molecular_vs_organic_benchmark.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from oneuro.organic_neural_network import (
    OrganicNeuralNetwork,
    XORTask,
    PatternRecognitionTask,
)
from oneuro.molecular.network import MolecularNeuralNetwork


def sparkline(values: list, width: int = 40) -> str:
    """ASCII sparkline of values."""
    if not values:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1.0
    chars = " ▁▂▃▄▅▆▇█"
    out = []
    # Downsample to width
    step = max(1, len(values) // width)
    for i in range(0, len(values), step):
        chunk = values[i:i + step]
        avg = sum(chunk) / len(chunk)
        idx = int((avg - mn) / rng * (len(chars) - 1))
        out.append(chars[idx])
    return "".join(out[:width])


def evaluate_network(network, task_cls, n_pre=5, n_train=200, n_post=20, seed=42):
    """Train and evaluate a network on a task.

    Returns dict with pre/post success rates, final reward, timing, and stats.
    """
    np.random.seed(seed)
    task = task_cls(network)

    # Pre-evaluate
    pre_results = network.evaluate_task(task, n_trials=n_pre)

    # Train
    t0 = time.time()
    np.random.seed(seed + 1)
    train_results = network.train_task(task, n_episodes=n_train, report_every=50)
    wall_time = time.time() - t0

    # Post-evaluate
    np.random.seed(seed + 2)
    post_results = network.evaluate_task(task, n_trials=n_post)

    # Learning curve
    curve = network.get_learning_curve(task.name)

    return {
        "task": task.name,
        "pre_success": pre_results["success_rate"],
        "post_success": post_results["success_rate"],
        "final_reward": train_results["final_avg_reward"],
        "neurons": train_results.get("total_neurons", len(network.neurons)),
        "synapses": train_results.get("total_synapses", len(getattr(network, "synapses", {}))),
        "wall_time": wall_time,
        "curve": curve,
    }


def run_benchmark():
    print("=" * 78)
    print("  oNeuro Benchmark: Molecular vs Organic Neural Networks")
    print("=" * 78)
    print()

    SIZE = (10, 10, 5)
    N_NEURONS = 25
    ENERGY = 3.0
    SEED = 42

    tasks = [XORTask, PatternRecognitionTask]
    results = []

    for task_cls in tasks:
        task_name = task_cls.__name__.replace("Task", "")
        print(f"--- {task_name} ---")
        print()

        for net_type in ["Organic", "Molecular"]:
            np.random.seed(SEED)
            if net_type == "Organic":
                network = OrganicNeuralNetwork(
                    size=SIZE, initial_neurons=N_NEURONS, energy_supply=ENERGY
                )
            else:
                network = MolecularNeuralNetwork(
                    size=SIZE, initial_neurons=N_NEURONS, energy_supply=ENERGY
                )

            r = evaluate_network(network, task_cls, seed=SEED)
            r["network_type"] = net_type
            results.append(r)

            improvement = r["post_success"] - r["pre_success"]
            print(f"  {net_type:10s}  pre={r['pre_success']:.0%}  post={r['post_success']:.0%}  "
                  f"Δ={improvement:+.0%}  reward={r['final_reward']:.2f}  "
                  f"n={r['neurons']}  s={r['synapses']}  t={r['wall_time']:.1f}s")
            print(f"  {'':10s}  curve: {sparkline(r['curve'])}")
            print()

    # Summary table
    print()
    print("=" * 78)
    print(f"{'Task':<20s} {'Network':<12s} {'Pre':>5s} {'Post':>5s} {'Δ':>6s} "
          f"{'Reward':>7s} {'Neurons':>7s} {'Synapses':>8s} {'Time':>6s}")
    print("-" * 78)
    for r in results:
        imp = r["post_success"] - r["pre_success"]
        print(f"{r['task']:<20s} {r['network_type']:<12s} "
              f"{r['pre_success']:>5.0%} {r['post_success']:>5.0%} {imp:>+5.0%} "
              f"{r['final_reward']:>7.2f} {r['neurons']:>7d} {r['synapses']:>8d} "
              f"{r['wall_time']:>5.1f}s")
    print("=" * 78)


if __name__ == "__main__":
    run_benchmark()
