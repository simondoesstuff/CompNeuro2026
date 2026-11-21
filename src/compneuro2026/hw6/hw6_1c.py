"""HW6 1(c): renewal-reward simulation of uncertain patch foraging."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


def intake(t, rho=0.5, p_high=0.5, travel=5):
    food = p_high * -np.expm1(-rho * np.asarray(t))
    return food / (food / rho + (1 - p_high) * t + travel)


def belief_rate(t, rho=0.5, p_high=0.5):
    high_survival = p_high * np.exp(-rho * np.asarray(t))
    return rho * high_survival / (high_survival + 1 - p_high)


def run(out, seed=4370):
    out.mkdir(parents=True, exist_ok=True)
    optimum = brentq(lambda t: intake(t) - belief_rate(t), 0, 100)
    known = 0.5 / (0.5 / 0.5 + 5)
    rng = np.random.default_rng(seed)
    n = 400_000
    # Common random draws make comparisons across waiting policies less noisy.
    high = rng.random(n) < 0.5
    discovery = rng.exponential(1 / 0.5, n)
    rows = []
    for wait in [2, optimum, 10]:
        found = high & (discovery <= wait)
        duration = np.where(found, discovery, wait) + 5
        measured = found.sum() / duration.sum()
        # Delta-method standard error of the renewal reward/time ratio.
        se = np.std(found - measured * duration, ddof=1) / (
            np.sqrt(n) * duration.mean()
        )
        rows.append({
            "wait": float(wait), "visits": n, "food": int(found.sum()),
            "total_time": float(duration.sum()), "measured_rate": float(measured),
            "theory_rate": float(intake(wait)), "standard_error": float(se),
            "error_in_standard_errors": float((measured - intake(wait)) / se),
        })
    result = {
        "seed": seed, "rho": 0.5, "p_high": 0.5, "travel": 5,
        "optimal_wait": optimum, "optimal_rate": float(intake(optimum)),
        "known_rate": known, "efficiency": float(intake(optimum) / known),
        "simulations": rows,
    }
    t = np.linspace(0, 16, 1000)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, intake(t), label=r"Intake $R(T)$")
    ax.plot(t, belief_rate(t), label=r"Belief rate $\widetilde{\lambda}(T)$")
    ax.axhline(known, ls="--", color="gray", label=r"Known patch type $R_{known}$")
    ax.axvline(optimum, ls=":", color="black", alpha=0.5)
    ax.plot(optimum, intake(optimum), "ko", zorder=5,
            label=f"Optimum: T={optimum:.4f}, R={intake(optimum):.5f}")
    ax.errorbar([r["wait"] for r in rows], [r["measured_rate"] for r in rows],
                yerr=[1.96 * r["standard_error"] for r in rows], fmt="s",
                capsize=4, label="Simulation (95% intervals)")
    ax.set(xlabel="Wait without food T (time units)", ylabel="Food per time unit",
           title="Uncertain patch foraging", ylim=(0, 0.26))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "1c_foraging.png", dpi=180)
    plt.close(fig)
    (out / "1c_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("hw6_results"))
    parser.add_argument("--seed", type=int, default=4370)
    args = parser.parse_args()
    run(args.output_dir, args.seed)
