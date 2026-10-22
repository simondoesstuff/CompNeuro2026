"""HW4 3(c): stability and asynchronous recall in Hebbian Hopfield networks."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr

N = 500
ALPHAS = np.arange(0.02, 0.621, 0.04)
N_CUES = 10


def hebbian_weights(patterns):
    patterns = patterns.astype(float, copy=False)
    weights = patterns.T @ patterns / patterns.shape[1]
    np.fill_diagonal(weights, 0.0)
    return weights


def recall(weights, pattern, rng, flip_fraction=0.1, max_sweeps=1000):
    state = pattern.copy()
    flipped = rng.choice(len(state), size=round(flip_fraction * len(state)), replace=False)
    state[flipped] *= -1
    fields = weights @ state
    for _ in range(max_sweeps):
        changed = False
        for index in rng.permutation(len(state)):
            field = fields[index]
            updated = 1 if field >= 0 else -1
            if updated != state[index]:
                change = updated - state[index]
                state[index] = updated
                fields += change * weights[:, index]
                changed = True
        if not changed:
            return float(np.dot(pattern.astype(float), state) / len(state)), True
    return float(np.dot(pattern.astype(float), state) / len(state)), False


def run(output_dir, seed=4370):
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for alpha in ALPHAS:
        n_patterns = max(1, round(alpha * N))
        patterns = rng.choice(np.array([-1, 1], dtype=np.int8), size=(n_patterns, N))
        weights = hebbian_weights(patterns.astype(float))
        fields = patterns @ weights
        aligned = patterns * fields
        unstable = float(np.mean(aligned < 0))
        theory = float(ndtr(-1 / np.sqrt(alpha)))
        overlaps, converged = [], []
        for cue_index in rng.integers(0, n_patterns, size=N_CUES):
            overlap, did_converge = recall(weights, patterns[cue_index].copy(), rng)
            overlaps.append(overlap)
            converged.append(did_converge)
        rows.append((alpha, n_patterns, unstable, theory, np.mean(overlaps),
                     np.std(overlaps, ddof=1), np.mean(converged)))
        print(f"alpha={alpha:.2f}, P={n_patterns}: unstable={unstable:.4f} "
              f"(theory {theory:.4f}), recall overlap={np.mean(overlaps):.4f} "
              f"+/- {np.std(overlaps, ddof=1):.4f}, converged={np.mean(converged):.0%}")

    values = np.asarray(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(values[:, 0], values[:, 2], "o-", label="Simulation")
    axes[0].plot(values[:, 0], values[:, 3], "--", label=r"$\Phi(-1/\sqrt{\alpha})$")
    axes[0].set(xlabel=r"Load $\alpha=P/N$", ylabel="Fraction of unstable stored bits")
    axes[1].errorbar(values[:, 0], values[:, 4], yerr=values[:, 5] / np.sqrt(N_CUES),
                     fmt="o-", capsize=3, label="Mean overlap (SEM)")
    axes[1].set(xlabel=r"Load $\alpha=P/N$", ylabel="Recall overlap $m$", ylim=(-0.05, 1.05))
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "3c_hopfield_capacity.png", dpi=180)
    plt.close(fig)
    np.savetxt(output_dir / "3c_summary.csv", values, delimiter=",")
    print("Columns: alpha, P, simulated_unstable_fraction, CLT_prediction, "
          "mean_recall_overlap, SD_recall_overlap, recall_convergence_fraction")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("hw4_results"))
    parser.add_argument("--seed", type=int, default=4370)
    args = parser.parse_args()
    run(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
