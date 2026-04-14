"""
Plot side-by-side comparison of workadam vs spikeprotect on 3-task continual learning.
Usage: python plot_comparison.py --workadam out/... --spikeprotect out/...
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_metrics(path):
    return pd.read_csv(os.path.join(path, "metrics.csv"))


def plot_comparison(workadam_dir, spikeprotect_dir, out_path="comparison.png"):
    df_w = load_metrics(workadam_dir)
    df_s = load_metrics(spikeprotect_dir)

    # Detect phases and boundaries from data
    phases = sorted(df_w["ndigit"].unique())
    phase_boundaries = []
    for i in range(len(phases) - 1):
        boundary = df_w[df_w["ndigit"] == phases[i]]["iter"].max()
        phase_boundaries.append(boundary)

    acc_cols = [c for c in df_w.columns if c.startswith("test_acc_")]
    digit_labels = [c.replace("test_acc_", "").replace("d", "-digit") for c in acc_cols]

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # --- Top row: accuracy curves ---
    for col_idx, (df, label, style) in enumerate([
        (df_w, "WorkAdam (original paper)", "-"),
        (df_s, "SpikeProtectAdam (ours)", "-"),
    ]):
        ax = fig.add_subplot(gs[0, col_idx])
        for i, (col, dlabel) in enumerate(zip(acc_cols, digit_labels)):
            ax.plot(df["iter"], df[col] * 100, color=colors[i],
                    linestyle=style, linewidth=1.8, label=dlabel)
        for b in phase_boundaries:
            ax.axvline(b, color="black", linestyle="--", alpha=0.5, linewidth=1.2,
                       label="task switch" if b == phase_boundaries[0] else "")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylim(-5, 105)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # --- Bottom left: overlay comparison at task boundaries ---
    ax = fig.add_subplot(gs[1, 0])
    for i, (col, dlabel) in enumerate(zip(acc_cols, digit_labels)):
        ax.plot(df_w["iter"], df_w[col] * 100, color=colors[i],
                linestyle="--", linewidth=1.5, alpha=0.7, label=f"WorkAdam {dlabel}")
        ax.plot(df_s["iter"], df_s[col] * 100, color=colors[i],
                linestyle="-", linewidth=1.8, label=f"SpikeProtect {dlabel}")
    for b in phase_boundaries:
        ax.axvline(b, color="black", linestyle=":", alpha=0.4, linewidth=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Overlay Comparison", fontsize=11, fontweight="bold")
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Bottom right: min accuracy at each task boundary ---
    ax = fig.add_subplot(gs[1, 1])
    boundary_labels = [f"Task {i+1}→{i+2}" for i in range(len(phase_boundaries))]
    x = range(len(phase_boundaries))
    width = 0.35

    for i, (col, dlabel) in enumerate(zip(acc_cols[:-1], digit_labels[:-1])):
        w_mins, s_mins = [], []
        for b in phase_boundaries:
            # accuracy of earlier task right after boundary (first 200 iters)
            w_after = df_w[(df_w["iter"] > b) & (df_w["iter"] <= b + 200)][col]
            s_after = df_s[(df_s["iter"] > b) & (df_s["iter"] <= b + 200)][col]
            w_mins.append(w_after.min() * 100 if len(w_after) > 0 else 0)
            s_mins.append(s_after.min() * 100 if len(s_after) > 0 else 0)

        offset = (i - len(acc_cols) / 2 + 0.5) * width
        bars_w = ax.bar([xi + offset - width/2 for xi in x], w_mins,
                        width=width * 0.9, color=colors[i], alpha=0.5, label=f"WorkAdam {dlabel}")
        bars_s = ax.bar([xi + offset + width/2 for xi in x], s_mins,
                        width=width * 0.9, color=colors[i], alpha=0.9, label=f"SpikeProtect {dlabel}",
                        edgecolor="black", linewidth=0.8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(boundary_labels)
    ax.set_ylabel("Min Accuracy at Boundary (%)")
    ax.set_title("Forgetting at Task Boundaries\n(higher = less forgetting)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Continual Learning: WorkAdam vs SpikeProtectAdam\n3-Task Setting (2→3→4 digit addition)",
                 fontsize=13, fontweight="bold", y=1.01)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workadam", required=True, help="Path to workadam metrics dir")
    parser.add_argument("--spikeprotect", required=True, help="Path to spikeprotect metrics dir")
    parser.add_argument("--out", default="comparison.png")
    args = parser.parse_args()
    plot_comparison(args.workadam, args.spikeprotect, args.out)
