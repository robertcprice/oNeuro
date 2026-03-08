#!/usr/bin/env python3
"""Generate publication-quality figures for the DishBrain replication paper.

Reads JSON results from demo_dishbrain_pong.py --json output and produces
6 figures suitable for journal submission (Neuron, Nature Machine Intelligence).

Usage:
    # From JSON data directory:
    python3 scripts/generate_paper_figures.py --data-dir results/

    # Using hardcoded A100 medium-scale data:
    python3 scripts/generate_paper_figures.py --manual

    # PDF vector output:
    python3 scripts/generate_paper_figures.py --manual --format pdf

    # Custom output directory:
    python3 scripts/generate_paper_figures.py --manual --output-dir figs/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Publication style configuration
# ---------------------------------------------------------------------------

STYLE_CONFIG = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "normal",
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "legend.fontsize": 9,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.8,
}

# Color palette -- colorblind-safe, print-friendly
COLORS = {
    "fep": "#2166AC",          # steel blue
    "da": "#B2182B",           # brick red
    "random": "#999999",       # medium gray
    "baseline": "#888888",     # gray
    "caffeine": "#4DAF4A",     # green
    "diazepam": "#E41A1C",     # red
    "amphetamine": "#377EB8",  # blue
    "methamphetamine": "#984EA3",  # purple
    "scale_1k": "#E66101",     # orange
    "scale_5k": "#5E3C99",     # violet
    "scale_10k": "#1B7837",    # forest green
    "scale_25k": "#D95F02",    # dark orange
    "scale_100k": "#7570B3",   # slate
}


# ---------------------------------------------------------------------------
# Hardcoded A100 medium-scale data (--manual mode)
# ---------------------------------------------------------------------------

MANUAL_DATA = {
    # Experiment 1: Pong learning curves (5050 neurons, seed 42)
    # Reconstructed from A100 run outputs -- 10-rally moving average snapshots
    "exp1_pong_outcomes_5k": {
        "rally_hitrate_moving_avg": [
            # (rally_number, hit_rate_10_rally_ma) -- approximated from run logs
            (10, 0.40), (20, 0.50), (30, 0.45), (40, 0.55),
            (50, 0.50), (60, 0.60), (70, 0.55), (80, 0.60),
        ],
        "first_10": 0.40,
        "last_10": 0.60,
        "n_neurons": 5050,
    },
    "exp1_pong_outcomes_1k": {
        "rally_hitrate_moving_avg": [
            (10, 0.30), (20, 0.40), (30, 0.50), (40, 0.60),
            (50, 0.60), (60, 0.70), (70, 0.60), (80, 0.70),
        ],
        "first_10": 0.30,
        "last_10": 0.70,
        "n_neurons": 1010,
    },

    # Experiment 2: FEP vs DA vs Random (3 seeds, 5050 neurons)
    "exp2_protocol_comparison": {
        "fep_total_hits": [37, 46, 35],      # seeds 42, 43, 44
        "da_total_hits": [29, 42, 36],
        "random_total_hits": [33, 30, 31],
        "n_rallies": 80,
    },

    # Experiment 3: Drug effects on Pong (medium scale, 3 seeds on A100)
    "exp3_pong_drugs": {
        "baseline": [9, 14, 15],          # seeds 42, 43, 44
        "caffeine": [13, 15, 16],
        "diazepam": [10, 12, 19],
        "amphetamine": [14, 11, 14],
        "methamphetamine": [11, 9, 13],
        "n_test_rallies": 30,
    },

    # Experiment 4: Spatial Arena navigation (5050 neurons)
    "exp4_arena_navigation": {
        "quarter_success_rates": [0.23, 0.31, 0.38, 0.46],
        "quarter_labels": ["Q1\n(ep 1-12)", "Q2\n(ep 13-25)",
                           "Q3\n(ep 26-37)", "Q4\n(ep 38-50)"],
    },

    # Experiment 5 (Spatial Arena drug effects): 5050 neurons, 3 seeds (42, 1042, 2042)
    "exp5_doom_drugs": {
        "conditions": ["Baseline", "Caffeine", "Diazepam", "Amphetamine", "Meth"],
        "scores": [[-21.5, -27.35, -18.2], [-15.85, -20.05, -21.35],
                   [-23.75, -12.35, -16.9], [-15.75, -22.35, -23.15],
                   [-14.95, -24.85, -24.55]],
        "damage": [[42, 49, 32], [32, 40, 42], [45, 27, 34], [42, 45, 44], [39, 40, 44]],
    },

    # Experiment 5 (Scale invariance): seed 43, multiple scales
    "exp6_scale_invariance": {
        "1K": {
            "rally_hitrate_moving_avg": [
                (10, 0.70), (20, 0.65), (30, 0.60), (40, 0.65),
                (50, 0.60), (60, 0.60),
            ],
            "n_neurons": 1010,
        },
        "5K": {
            "rally_hitrate_moving_avg": [
                (10, 0.40), (20, 0.45), (30, 0.50), (40, 0.45),
                (50, 0.50), (60, 0.50),
            ],
            "n_neurons": 5050,
        },
        "10K": {
            "rally_hitrate_moving_avg": [
                (10, 0.60), (20, 0.65), (30, 0.60), (40, 0.65),
                (50, 0.70), (60, 0.70),
            ],
            "n_neurons": 10100,
        },
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json_results(data_dir: str) -> Dict[str, Any]:
    """Load all JSON result files from a directory.

    Expects files produced by:
        python3 demos/demo_dishbrain_pong.py --json results.json

    Returns a merged dictionary keyed by experiment number.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        print(f"ERROR: data directory not found: {data_dir}")
        sys.exit(1)

    merged: Dict[str, Any] = {}
    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        print(f"ERROR: no JSON files found in {data_dir}")
        sys.exit(1)

    for jf in json_files:
        print(f"  Loading: {jf.name}")
        with open(jf) as f:
            data = json.load(f)

        # The demo outputs a top-level structure with "runs" containing experiments
        if "runs" in data:
            for run in data["runs"]:
                seed = run.get("seed", "unknown")
                for exp_id, exp_data in run.get("experiments", {}).items():
                    key = f"exp{exp_id}_seed{seed}"
                    merged[key] = exp_data
                    merged[key]["_source_file"] = jf.name
                    merged[key]["_seed"] = seed
                    merged[key]["_scale"] = data.get("scale", "unknown")

        # Also store raw data for multi-run aggregation
        fname = jf.stem
        merged[f"_raw_{fname}"] = data

    print(f"  Loaded {len(json_files)} file(s), {len(merged)} result entries")
    return merged


def _moving_average(values: List[int], window: int = 10) -> List[float]:
    """Compute moving average over a binary outcome list."""
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start:i + 1]
        result.append(sum(chunk) / len(chunk))
    return result


def _extract_outcomes_from_json(
    merged: Dict[str, Any], exp_num: str
) -> Dict[str, List[int]]:
    """Extract outcome lists for a given experiment across seeds."""
    outcomes = {}
    for key, val in merged.items():
        if key.startswith(f"exp{exp_num}_seed") and "outcomes" in val:
            seed = val.get("_seed", key)
            outcomes[str(seed)] = val["outcomes"]
        elif key.startswith(f"exp{exp_num}_seed") and "all_outcomes" in val:
            for condition, oc in val["all_outcomes"].items():
                ckey = f"{condition}_seed{val.get('_seed', key)}"
                outcomes[ckey] = oc
    return outcomes


# ---------------------------------------------------------------------------
# Figure 1: Pong Learning Curve
# ---------------------------------------------------------------------------

def figure1_pong_learning(
    merged: Optional[Dict[str, Any]],
    manual: bool,
    output_dir: Path,
    fmt: str,
) -> None:
    """Pong hit rate over rallies with 10-rally moving average."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    if manual:
        # Plot medium-scale (5K) curve
        data_5k = MANUAL_DATA["exp1_pong_outcomes_5k"]
        rallies_5k = [p[0] for p in data_5k["rally_hitrate_moving_avg"]]
        rates_5k = [p[1] for p in data_5k["rally_hitrate_moving_avg"]]
        ax.plot(rallies_5k, rates_5k, "o-", color=COLORS["fep"],
                label=f"5K neurons ({data_5k['n_neurons']})", markersize=6)

        # Plot small-scale (1K) curve
        data_1k = MANUAL_DATA["exp1_pong_outcomes_1k"]
        rallies_1k = [p[0] for p in data_1k["rally_hitrate_moving_avg"]]
        rates_1k = [p[1] for p in data_1k["rally_hitrate_moving_avg"]]
        ax.plot(rallies_1k, rates_1k, "s--", color=COLORS["scale_1k"],
                label=f"1K neurons ({data_1k['n_neurons']})", markersize=5)

    else:
        # From JSON: look for exp1 outcomes
        for key, val in merged.items():
            if key.startswith("exp1_seed") and "outcomes" in val:
                outcomes = val["outcomes"]
                ma = _moving_average(outcomes, window=10)
                seed = val.get("_seed", "?")
                scale = val.get("_scale", "?")
                ax.plot(range(1, len(ma) + 1), ma,
                        label=f"seed {seed} ({scale})", alpha=0.85)

    # Random baseline
    ax.axhline(y=0.30, color=COLORS["random"], linestyle="--",
               linewidth=1.5, label="Random baseline (30%)", zorder=1)

    ax.set_xlabel("Rally Number")
    ax.set_ylabel("Hit Rate (10-rally moving average)")
    ax.set_title("DishBrain Pong Replication \u2014 Free Energy Learning")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, 85)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, axis="y")

    _save(fig, output_dir / f"fig1_pong_learning.{fmt}", fmt)


# ---------------------------------------------------------------------------
# Figure 2: FEP vs DA vs Random
# ---------------------------------------------------------------------------

def figure2_protocol_comparison(
    merged: Optional[Dict[str, Any]],
    manual: bool,
    output_dir: Path,
    fmt: str,
) -> None:
    """Bar chart comparing learning protocols."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    if manual:
        data = MANUAL_DATA["exp2_protocol_comparison"]
        fep = np.array(data["fep_total_hits"], dtype=float)
        da = np.array(data["da_total_hits"], dtype=float)
        rand = np.array(data["random_total_hits"], dtype=float)
        n_rallies = data["n_rallies"]

        means = [fep.mean(), da.mean(), rand.mean()]
        sems = [fep.std(ddof=1) / np.sqrt(len(fep)),
                da.std(ddof=1) / np.sqrt(len(da)),
                rand.std(ddof=1) / np.sqrt(len(rand))]
        labels = ["Free Energy\n(FEP)", "DA Reward", "Random\n(Control)"]
        colors = [COLORS["fep"], COLORS["da"], COLORS["random"]]

    else:
        # Extract from JSON: exp2 has "results" with condition totals
        fep_totals, da_totals, rand_totals = [], [], []
        for key, val in merged.items():
            if key.startswith("exp2_seed"):
                results = val.get("results", {})
                if "free_energy" in results:
                    fep_totals.append(results["free_energy"]["total"])
                if "da_reward" in results:
                    da_totals.append(results["da_reward"]["total"])
                if "random" in results:
                    rand_totals.append(results["random"]["total"])

        if not fep_totals:
            print("  WARNING: No Exp 2 data found in JSON, skipping Figure 2")
            plt.close(fig)
            return

        fep = np.array(fep_totals, dtype=float)
        da = np.array(da_totals, dtype=float)
        rand = np.array(rand_totals, dtype=float)

        means = [fep.mean(), da.mean(), rand.mean()]
        sems = [
            fep.std(ddof=1) / np.sqrt(len(fep)) if len(fep) > 1 else 0,
            da.std(ddof=1) / np.sqrt(len(da)) if len(da) > 1 else 0,
            rand.std(ddof=1) / np.sqrt(len(rand)) if len(rand) > 1 else 0,
        ]
        labels = ["Free Energy\n(FEP)", "DA Reward", "Random\n(Control)"]
        colors = [COLORS["fep"], COLORS["da"], COLORS["random"]]

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=sems, width=0.55, color=colors,
                  edgecolor="white", linewidth=1.2, capsize=5,
                  error_kw={"linewidth": 1.5, "capthick": 1.5})

    # Value annotations above bars
    for bar, mean, sem in zip(bars, means, sems):
        y_pos = bar.get_height() + sem + 0.8
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold")

    # Seed count annotation
    n_seeds = len(fep) if manual else len(fep_totals)
    if n_seeds > 1:
        ax.text(0.98, 0.02, f"n = {n_seeds} seeds\nerror bars = SEM",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="0.5", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Total Hits (out of 80 rallies)")
    ax.set_title("Learning Protocol Comparison")
    ax.set_ylim(0, max(means) * 1.35)
    ax.grid(True, axis="y")

    _save(fig, output_dir / f"fig2_protocol_comparison.{fmt}", fmt)


# ---------------------------------------------------------------------------
# Figure 3: Pharmacological Effects on Pong
# ---------------------------------------------------------------------------

def figure3_pong_drugs(
    merged: Optional[Dict[str, Any]],
    manual: bool,
    output_dir: Path,
    fmt: str,
) -> None:
    """Bar chart of drug effects on Pong test performance."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    conditions_order = ["baseline", "caffeine", "diazepam",
                        "amphetamine", "methamphetamine"]
    display_labels = ["Baseline", "Caffeine", "Diazepam",
                      "Amphetamine", "Meth-\namphetamine"]
    bar_colors = [COLORS["baseline"], COLORS["caffeine"], COLORS["diazepam"],
                  COLORS["amphetamine"], COLORS["methamphetamine"]]

    if manual:
        data = MANUAL_DATA["exp3_pong_drugs"]
        n_test = data["n_test_rallies"]
        hits_list = []
        valid_labels = []
        valid_colors = []
        sems = []
        for i, cond in enumerate(conditions_order):
            val = data.get(cond)
            if val is not None:
                if isinstance(val, list):
                    hits_list.append(np.mean(val))
                    sems.append(
                        np.std(val, ddof=1) / np.sqrt(len(val))
                        if len(val) > 1 else 0
                    )
                else:
                    hits_list.append(val)
                    sems.append(0)
                valid_labels.append(display_labels[i])
                valid_colors.append(bar_colors[i])

    else:
        # Extract from JSON: exp3 has "test_results" dict
        all_hits: Dict[str, List[int]] = {c: [] for c in conditions_order}
        for key, val in merged.items():
            if key.startswith("exp3_seed") and "test_results" in val:
                tr = val["test_results"]
                for cond in conditions_order:
                    if cond in tr:
                        all_hits[cond].append(tr[cond]["hits"])

        if not any(all_hits.values()):
            print("  WARNING: No Exp 3 data found in JSON, skipping Figure 3")
            plt.close(fig)
            return

        hits_list = []
        valid_labels = []
        valid_colors = []
        sems = []
        for i, cond in enumerate(conditions_order):
            vals = all_hits[cond]
            if vals:
                hits_list.append(np.mean(vals))
                valid_labels.append(display_labels[i])
                valid_colors.append(bar_colors[i])
                sems.append(
                    np.std(vals, ddof=1) / np.sqrt(len(vals))
                    if len(vals) > 1 else 0
                )
        n_test = 30  # default

    x = np.arange(len(valid_labels))
    bar_kwargs = dict(
        width=0.55, color=valid_colors, edgecolor="white", linewidth=1.2,
    )
    if sems and any(s > 0 for s in sems):
        bar_kwargs["yerr"] = sems
        bar_kwargs["capsize"] = 5
        bar_kwargs["error_kw"] = {"linewidth": 1.5, "capthick": 1.5}

    bars = ax.bar(x, hits_list, **bar_kwargs)

    # Value annotations
    for bar, val in zip(bars, hits_list):
        y_pos = bar.get_height() + 0.3
        label = f"{val:.0f}" if isinstance(val, (int, np.integer)) else f"{val:.1f}"
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(valid_labels)
    ax.set_ylabel(f"Test Hits (out of {n_test})")
    ax.set_title("Pharmacological Modulation of Game Performance")
    ax.set_ylim(0, max(hits_list) * 1.3 if hits_list else 20)
    ax.grid(True, axis="y")

    _save(fig, output_dir / f"fig3_pong_drugs.{fmt}", fmt)


# ---------------------------------------------------------------------------
# Figure 4: Spatial Arena Navigation
# ---------------------------------------------------------------------------

def figure4_arena_navigation(
    merged: Optional[Dict[str, Any]],
    manual: bool,
    output_dir: Path,
    fmt: str,
) -> None:
    """Quarter-by-quarter success rate for spatial navigation."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    if manual:
        data = MANUAL_DATA["exp4_arena_navigation"]
        rates = data["quarter_success_rates"]
        labels = data["quarter_labels"]

    else:
        # Extract from JSON: exp4 has "steps_per_episode"
        all_steps = []
        max_steps = 40  # default from SimpleArena
        for key, val in merged.items():
            if key.startswith("exp4_seed") and "steps_per_episode" in val:
                all_steps.append(val["steps_per_episode"])

        if not all_steps:
            print("  WARNING: No Exp 4 data found in JSON, skipping Figure 4")
            plt.close(fig)
            return

        # Average across seeds, then compute quarter success rates
        max_len = max(len(s) for s in all_steps)
        # Pad shorter runs and compute per-quarter success
        quarter_size = max_len // 4
        rates = []
        labels = []
        for q in range(4):
            start = q * quarter_size
            end = start + quarter_size
            successes = 0
            total = 0
            for steps in all_steps:
                chunk = steps[start:min(end, len(steps))]
                successes += sum(1 for s in chunk if s < max_steps)
                total += len(chunk)
            rates.append(successes / total if total > 0 else 0)
            labels.append(f"Q{q+1}\n(ep {start+1}-{end})")

    x = np.arange(len(rates))

    # Combined: bar chart with overlaid line
    bars = ax.bar(x, rates, width=0.5, color=COLORS["fep"], alpha=0.6,
                  edgecolor=COLORS["fep"], linewidth=1.2)
    ax.plot(x, rates, "o-", color=COLORS["fep"], markersize=8,
            linewidth=2.0, zorder=5)

    # Value annotations
    for xi, rate in zip(x, rates):
        ax.text(xi, rate + 0.02, f"{rate:.0%}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=COLORS["fep"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Success Rate (reached target)")
    ax.set_title("Spatial Navigation Learning")
    ax.set_ylim(0, min(1.0, max(rates) * 1.3) if rates else 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(True, axis="y")

    # Trend annotation
    if len(rates) >= 2 and rates[-1] > rates[0]:
        improvement = rates[-1] - rates[0]
        ax.annotate(
            f"+{improvement:.0%} improvement",
            xy=(len(rates) - 1, rates[-1]),
            xytext=(len(rates) - 1.8, rates[-1] + 0.08),
            fontsize=9, color="0.3", style="italic",
            arrowprops=dict(arrowstyle="->", color="0.5", lw=1.2),
        )

    _save(fig, output_dir / f"fig4_arena_navigation.{fmt}", fmt)


# ---------------------------------------------------------------------------
# Figure 5: Spatial Arena Drug Effects (dual Y-axis)
# ---------------------------------------------------------------------------

def figure5_doom_drugs(
    merged: Optional[Dict[str, Any]],
    manual: bool,
    output_dir: Path,
    fmt: str,
) -> None:
    """Grouped bar chart with dual Y-axis: score (left) and damage (right)."""
    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    if manual:
        data = MANUAL_DATA["exp5_doom_drugs"]
        conditions = data["conditions"]
        raw_scores = data["scores"]
        raw_damage = data["damage"]
        # Support multi-seed lists: compute means and SEMs
        if isinstance(raw_scores[0], list):
            scores = [np.mean(s) for s in raw_scores]
            damage = [np.mean(d) for d in raw_damage]
            score_sems = [np.std(s) / np.sqrt(len(s)) for s in raw_scores]
            damage_sems = [np.std(d) / np.sqrt(len(d)) for d in raw_damage]
        else:
            scores = raw_scores
            damage = raw_damage
            score_sems = None
            damage_sems = None

    else:
        # Spatial Arena drug data would be in a separate experiment JSON
        # Look for doom-specific keys or exp5 keys with score/damage
        conditions, scores, damage = [], [], []
        score_sems, damage_sems = None, None
        for key, val in merged.items():
            if "doom" in key.lower() or (
                key.startswith("exp5_") and "scores" in val
            ):
                conditions = val.get("conditions", [])
                scores = val.get("scores", [])
                damage = val.get("damage", [])
                break

        # Fallback: look for doom results inside exp3-style structures
        if not conditions:
            for key, val in merged.items():
                if key.startswith("exp3_seed") and "doom_results" in val:
                    dr = val["doom_results"]
                    conditions = list(dr.keys())
                    scores = [dr[c].get("score", 0) for c in conditions]
                    damage = [dr[c].get("damage", 0) for c in conditions]
                    break

        if not conditions:
            print("  WARNING: No Spatial Arena drug data found in JSON, skipping Figure 5")
            plt.close(fig)
            return

    x = np.arange(len(conditions))
    bar_width = 0.35

    # Condition colors
    cond_colors_score = []
    cond_colors_damage = []
    for cond in conditions:
        cond_lower = cond.lower()
        if "caffeine" in cond_lower:
            cond_colors_score.append(COLORS["caffeine"])
            cond_colors_damage.append(_lighten(COLORS["caffeine"], 0.4))
        elif "diazepam" in cond_lower:
            cond_colors_score.append(COLORS["diazepam"])
            cond_colors_damage.append(_lighten(COLORS["diazepam"], 0.4))
        elif "amphetamine" in cond_lower and "meth" not in cond_lower:
            cond_colors_score.append(COLORS["amphetamine"])
            cond_colors_damage.append(_lighten(COLORS["amphetamine"], 0.4))
        elif "meth" in cond_lower:
            cond_colors_score.append(COLORS["methamphetamine"])
            cond_colors_damage.append(_lighten(COLORS["methamphetamine"], 0.4))
        else:
            cond_colors_score.append(COLORS["baseline"])
            cond_colors_damage.append(_lighten(COLORS["baseline"], 0.4))

    # Left axis: Score (with error bars if available)
    bar_kwargs_score = dict(color=cond_colors_score, edgecolor="white",
                            linewidth=1.0, label="Score")
    if manual and score_sems is not None:
        bar_kwargs_score["yerr"] = score_sems
        bar_kwargs_score["capsize"] = 3
        bar_kwargs_score["error_kw"] = {"elinewidth": 1.2}
    bars1 = ax1.bar(x - bar_width / 2, scores, bar_width, **bar_kwargs_score)
    ax1.set_ylabel("Score", color=COLORS["fep"])
    ax1.tick_params(axis="y", labelcolor=COLORS["fep"])

    # Right axis: Damage (with error bars if available)
    bar_kwargs_damage = dict(color=cond_colors_damage, edgecolor="white",
                             linewidth=1.0, label="Damage taken", hatch="//",
                             alpha=0.85)
    if manual and damage_sems is not None:
        bar_kwargs_damage["yerr"] = damage_sems
        bar_kwargs_damage["capsize"] = 3
        bar_kwargs_damage["error_kw"] = {"elinewidth": 1.2}
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + bar_width / 2, damage, bar_width, **bar_kwargs_damage)
    ax2.set_ylabel("Damage Taken", color=COLORS["da"])
    ax2.tick_params(axis="y", labelcolor=COLORS["da"])
    ax2.spines["right"].set_visible(True)

    # Value annotations -- place score labels inside bars (midpoint) to avoid
    # overlap with x-axis labels on negative bars
    for bar, val in zip(bars1, scores):
        bar_mid = bar.get_height() / 2
        ax1.text(bar.get_x() + bar.get_width() / 2, bar_mid,
                 f"{val:.1f}", ha="center", va="center", fontsize=8,
                 fontweight="bold", color="white")

    for bar, val in zip(bars2, damage):
        bar_mid = bar.get_height() / 2
        ax2.text(bar.get_x() + bar.get_width() / 2, bar_mid,
                 f"{val}", ha="center", va="center", fontsize=8,
                 fontweight="bold", color="0.25")

    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions)
    ax1.set_title("Pharmacological Effects on Spatial Navigation")

    # Combined legend -- place above the plot to avoid bar overlap
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper center", ncol=2, frameon=True,
               bbox_to_anchor=(0.5, -0.12))

    ax1.grid(True, axis="y", alpha=0.2)
    fig.subplots_adjust(bottom=0.2)

    _save(fig, output_dir / f"fig5_doom_drugs.{fmt}", fmt)


# ---------------------------------------------------------------------------
# Figure 6: Scale Invariance
# ---------------------------------------------------------------------------

def figure6_scale_invariance(
    merged: Optional[Dict[str, Any]],
    manual: bool,
    output_dir: Path,
    fmt: str,
) -> None:
    """Overlaid learning curves for multiple network scales."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    scale_colors = {
        "1K": COLORS["scale_1k"],
        "5K": COLORS["scale_5k"],
        "10K": COLORS["scale_10k"],
        "25K": COLORS["scale_25k"],
        "100K": COLORS["scale_100k"],
    }
    scale_markers = {"1K": "o", "5K": "s", "10K": "^", "25K": "D", "100K": "v"}

    if manual:
        data = MANUAL_DATA["exp6_scale_invariance"]
        for tier_name, tier_data in data.items():
            rallies = [p[0] for p in tier_data["rally_hitrate_moving_avg"]]
            rates = [p[1] for p in tier_data["rally_hitrate_moving_avg"]]
            color = scale_colors.get(tier_name, "#333333")
            marker = scale_markers.get(tier_name, "o")
            ax.plot(rallies, rates, f"{marker}-", color=color,
                    label=f"{tier_name} ({tier_data['n_neurons']} neurons)",
                    markersize=6, linewidth=2.0)

    else:
        # Extract from JSON: exp5 has "tier_results"
        for key, val in merged.items():
            if key.startswith("exp5_seed") and "tier_results" in val:
                for tier_name, tier_data in val["tier_results"].items():
                    # If there are raw outcomes, compute moving average
                    # Otherwise use first_10/last_10 as endpoints
                    color = scale_colors.get(tier_name, "#333333")
                    marker = scale_markers.get(tier_name, "o")
                    n_neurons = tier_data.get("n_neurons", "?")

                    if "outcomes" in tier_data:
                        outcomes = tier_data["outcomes"]
                        ma = _moving_average(outcomes, window=10)
                        ax.plot(range(1, len(ma) + 1), ma,
                                f"{marker}-", color=color,
                                label=f"{tier_name} ({n_neurons} neurons)",
                                markersize=4, linewidth=1.8, alpha=0.85)
                    else:
                        # Only have summary stats -- plot two endpoints
                        f10 = tier_data.get("first_10", 0.3)
                        l10 = tier_data.get("last_10", 0.5)
                        total = tier_data.get("total", 0.4)
                        ax.plot([10, 50], [f10, l10],
                                f"{marker}--", color=color,
                                label=f"{tier_name} ({n_neurons})",
                                markersize=8, linewidth=2.0)

    # Random baseline
    ax.axhline(y=0.30, color=COLORS["random"], linestyle="--",
               linewidth=1.5, label="Random baseline (30%)", zorder=1)

    ax.set_xlabel("Rally Number")
    ax.set_ylabel("Hit Rate (10-rally moving average)")
    ax.set_title("Scale Invariance of FEP Learning")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, axis="y")

    _save(fig, output_dir / f"fig6_scale_invariance.{fmt}", fmt)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _lighten(hex_color: str, amount: float = 0.3) -> str:
    """Lighten a hex color by blending toward white."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def _save(fig: plt.Figure, path: Path, fmt: str) -> None:
    """Save figure and report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), format=fmt, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    size_kb = path.stat().st_size / 1024
    print(f"  Saved: {path} ({size_kb:.0f} KB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for the DishBrain "
                    "replication paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 scripts/generate_paper_figures.py --manual\n"
            "  python3 scripts/generate_paper_figures.py --data-dir results/\n"
            "  python3 scripts/generate_paper_figures.py --manual --format pdf\n"
        ),
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Directory containing JSON result files from "
             "demo_dishbrain_pong.py --json",
    )
    parser.add_argument(
        "--manual", action="store_true",
        help="Use hardcoded A100 medium-scale data instead of JSON files",
    )
    parser.add_argument(
        "--format", type=str, default="png", choices=["png", "pdf", "svg"],
        help="Output format: png (raster, 300 DPI) or pdf/svg (vector). "
             "Default: png",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for figures. Default: papers/figures/",
    )
    parser.add_argument(
        "--figures", type=int, nargs="*", default=None,
        help="Which figures to generate (1-6). Default: all",
    )

    args = parser.parse_args()

    if not args.manual and not args.data_dir:
        parser.error("Specify --data-dir PATH or --manual")

    # Resolve output directory
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "papers" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Apply publication style
    plt.rcParams.update(STYLE_CONFIG)

    print("=" * 60)
    print("  DishBrain Replication Paper -- Figure Generation")
    print(f"  Mode:   {'manual (hardcoded A100 data)' if args.manual else 'JSON'}")
    print(f"  Format: {args.format.upper()} ({'raster 300 DPI' if args.format == 'png' else 'vector'})")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    # Load data
    merged = None
    if not args.manual:
        merged = load_json_results(args.data_dir)

    # Which figures to generate
    figure_ids = args.figures if args.figures else [1, 2, 3, 4, 5, 6]

    figure_funcs = {
        1: ("Pong Learning Curve", figure1_pong_learning),
        2: ("FEP vs DA vs Random", figure2_protocol_comparison),
        3: ("Pharmacological Effects on Pong", figure3_pong_drugs),
        4: ("Spatial Arena Navigation", figure4_arena_navigation),
        5: ("Spatial Arena Drug Effects", figure5_doom_drugs),
        6: ("Scale Invariance", figure6_scale_invariance),
    }

    generated = 0
    for fig_id in figure_ids:
        if fig_id not in figure_funcs:
            print(f"\n  Unknown figure: {fig_id}")
            continue
        name, func = figure_funcs[fig_id]
        print(f"\n  Figure {fig_id}: {name}")
        try:
            func(merged, args.manual, output_dir, args.format)
            generated += 1
        except Exception as e:
            print(f"  ERROR generating Figure {fig_id}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"  Generated {generated}/{len(figure_ids)} figures in {output_dir}")
    print(f"{'=' * 60}")

    return 0 if generated == len(figure_ids) else 1


if __name__ == "__main__":
    sys.exit(main())
