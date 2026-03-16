"""
progress.py — Visualize autoresearch experiment progress.
Reads results.tsv, generates progress.png and prints text summary.

Usage:
    uv run progress.py
"""

import sys
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    tsv_path = "results.tsv"
    if not os.path.exists(tsv_path):
        print("No results.tsv found. Run some experiments first.")
        sys.exit(0)

    df = pd.read_csv(tsv_path, sep="\t")
    if len(df) == 0:
        print("results.tsv is empty.")
        sys.exit(0)

    df["tok_sec"] = pd.to_numeric(df["tok_sec"], errors="coerce")
    df["params_B"] = pd.to_numeric(df["params_B"], errors="coerce")
    df["active_B"] = pd.to_numeric(df["active_B"], errors="coerce")
    df["mem_gb"] = pd.to_numeric(df["mem_gb"], errors="coerce")
    df["status"] = df["status"].str.strip().str.lower()

    n_total = len(df)
    n_keep = len(df[df["status"] == "keep"])
    n_discard = len(df[df["status"] == "discard"])
    n_crash = len(df[df["status"] == "crash"])

    # Text summary
    print(f"\n=== Autoresearch Progress ===")
    print(f"Experiments: {n_total} ({n_keep} keep, {n_discard} discard, {n_crash} crash)")

    kept = df[df["status"] == "keep"]
    if len(kept) > 0:
        best_row = kept.loc[kept["params_B"].idxmax()]
        print(f"Biggest kept: {best_row['model']} — {best_row['params_B']:.0f}B params, "
              f"{best_row.get('active_B', '?')}B active, {best_row['tok_sec']:.1f} tok/s, "
              f"{best_row['mem_gb']:.1f} GB peak")

        fastest_row = kept.loc[kept["tok_sec"].idxmax()]
        print(f"Fastest kept: {fastest_row['model']} — {fastest_row['tok_sec']:.1f} tok/s, "
              f"{fastest_row['params_B']:.0f}B params")

    print(f"\nAll experiments:")
    for i, row in df.iterrows():
        status_icon = {"keep": "+", "discard": "-", "crash": "X"}.get(row["status"], "?")
        print(f"  [{status_icon}] #{i+1:3d}  {row['model']:40s}  "
              f"{row['tok_sec']:6.1f} tok/s  {row['mem_gb']:5.1f}GB  {row['description']}")

    # Plot
    if n_total < 2:
        print("\nNeed at least 2 experiments for a chart.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    x = range(1, n_total + 1)

    # Top: tok/sec by experiment
    colors = {"keep": "#2ecc71", "discard": "#95a5a6", "crash": "#e74c3c"}
    for status in ["discard", "crash", "keep"]:
        mask = df["status"] == status
        if mask.any():
            ax1.scatter([i+1 for i in df.index[mask]], df.loc[mask, "tok_sec"],
                       c=colors.get(status, "#333"), s=60 if status == "keep" else 25,
                       label=status.title(), zorder=4 if status == "keep" else 2,
                       edgecolors="black" if status == "keep" else "none",
                       linewidths=0.5)

    # Running best tok/sec for kept experiments
    if n_keep > 1:
        kept_sorted = df[df["status"] == "keep"].copy()
        running_best = kept_sorted["tok_sec"].cummax()
        ax1.step([i+1 for i in kept_sorted.index], running_best,
                where="post", color="#27ae60", linewidth=2, alpha=0.7, label="Running best")

    ax1.set_ylabel("tok/sec (higher is better)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.2)
    ax1.set_title(f"Autoresearch Progress: {n_total} Experiments", fontsize=13)

    # Bottom: model size
    for status in ["discard", "crash", "keep"]:
        mask = df["status"] == status
        if mask.any():
            ax2.scatter([i+1 for i in df.index[mask]], df.loc[mask, "params_B"],
                       c=colors.get(status, "#333"), s=60 if status == "keep" else 25,
                       zorder=4 if status == "keep" else 2,
                       edgecolors="black" if status == "keep" else "none",
                       linewidths=0.5)

    ax2.set_ylabel("Model Size (B params)", fontsize=11)
    ax2.set_xlabel("Experiment #", fontsize=11)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("progress.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved progress.png")


if __name__ == "__main__":
    main()
