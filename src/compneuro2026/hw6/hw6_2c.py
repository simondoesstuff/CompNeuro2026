"""HW6 2(c): stochastic policy gradients with and without entropy."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit

EQUILIBRIUM = np.array([8 / 17, 9 / 17])
# Rows: shoot 2, shoot 3; columns: guard D3, guard D2 (action 0, 1).
MAKE_PROBABILITY = np.array([[0.60, 0.40], [0.25, 0.40]])


def invariant(xy):
    """Closed-form definite integrals in K, with the last axis indexing x,y."""
    xy = np.asarray(xy)

    def primitive(s):
        return ((1 - 2 * EQUILIBRIUM) * (np.log(s) - np.log1p(-s))
                + EQUILIBRIUM / s + (1 - EQUILIBRIUM) / (1 - s))

    return np.sum(primitive(xy) - primitive(EQUILIBRIUM), axis=-1)


def mean_flow(rtol=1e-10):
    # eta=1 rescales time; at eta=.02 this interval is 50,000 possessions.
    def rhs(t, xy):
        x, y = xy
        return [0.85 * (x * (1 - x))**2 * (y - EQUILIBRIUM[1]),
                -0.85 * (y * (1 - y))**2 * (x - EQUILIBRIUM[0])]

    sol = solve_ivp(rhs, (0, 1000), [0.2, 0.3], method="DOP853",
                    rtol=rtol, atol=rtol * 0.01, dense_output=True, max_step=1)
    if not sol.success:
        raise RuntimeError(sol.message)
    times = np.linspace(0, 1000, 20_001)
    return times, sol.sol(times).T


def simulate(tau, seed=4370, n=400_000, eta=0.02):
    rng = np.random.default_rng(seed)
    # Pre-drawn uniforms allow the two bonus strengths to use common randomness.
    uniforms = rng.random((n, 3))
    theta = np.array([0.3, -0.3])
    trajectory = np.empty((n + 1, 2))
    trajectory[0] = expit(theta)
    baseline = 0.0
    for k, (u_offense, u_defense, u_shot) in enumerate(uniforms):
        p = expit(theta)
        a1, a2 = int(u_offense < p[0]), int(u_defense < p[1])
        points = (2 + a1) * (u_shot < MAKE_PROBABILITY[a1, a2])
        # Both updates use pre-possession policies and the previous baseline.
        advantage = points - baseline
        gradient = np.array([advantage * (a1 - p[0]),
                             -advantage * (a2 - p[1])])
        theta += eta * (gradient - tau * theta * p * (1 - p))
        baseline += (points - baseline) / (k + 1)
        trajectory[k + 1] = expit(theta)
    samples = trajectory[1:]
    quarter = n // 4
    distances = np.linalg.norm(samples - EQUILIBRIUM, axis=1)
    result = {
        "tau": tau, "seed": seed, "possessions": n, "eta": eta,
        "initial_logits": [0.3, -0.3], "final_baseline_points": float(baseline),
        "last_quarter_average": samples[-quarter:].mean(axis=0).tolist(),
        "early_mean_distance": float(distances[:quarter].mean()),
        "late_mean_distance": float(distances[-quarter:].mean()),
        "early_window": [1, quarter], "late_window": [n - quarter + 1, n],
        "final_probabilities": samples[-1].tolist(),
    }
    return trajectory, result


def run(out, seed=4370):
    out.mkdir(parents=True, exist_ok=True)
    times, orbit = mean_flow()
    k = invariant(orbit)
    flow_result = {
        "eta": 1, "time_interval": [0, 1000], "initial_probabilities": [0.2, 0.3],
        "x_range": [float(orbit[:, 0].min()), float(orbit[:, 0].max())],
        "y_range": [float(orbit[:, 1].min()), float(orbit[:, 1].max())],
        "initial_K": float(k[0]), "K_variation": float(np.ptp(k)),
        "max_absolute_K_drift": float(np.max(abs(k - k[0]))),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(*orbit.T)
    axes[0].plot(*orbit[0], "o", label="Start (0.2, 0.3)")
    axes[0].plot(*EQUILIBRIUM, "k*", ms=12, label="Nash equilibrium")
    axes[0].set(xlabel="x: shoot 3 probability", ylabel="y: guard paint probability",
                title="Mean flow (eta=1)", aspect="equal")
    axes[0].legend(loc="center", bbox_to_anchor=(0.5, 0.32))
    axes[1].plot(times, k - k[0])
    axes[1].set(xlabel="Rescaled time", ylabel="K(t) - K(0)", title="Conservation error")
    fig.tight_layout()
    fig.savefig(out / "2c_mean_flow.png", dpi=180)
    plt.close(fig)
    np.savetxt(out / "2c_mean_flow.csv", np.column_stack([times, orbit, k]),
               delimiter=",", header="rescaled_time,x,y,K", comments="")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    rows = []
    for column, tau in enumerate([0, 0.05]):
        path, row = simulate(tau, seed)
        rows.append(row)
        ax = axes[0, column]
        # Save/plot every 100th state; metrics above use every possession.
        indices = np.arange(0, len(path), 100)
        ax.plot(*path[indices].T, lw=0.5, alpha=0.55)
        ax.plot(*path[0], "o", label="Start")
        ax.plot(*EQUILIBRIUM, "k*", ms=12, label="Nash equilibrium")
        ax.plot(*row["last_quarter_average"], "rs", label="Last-quarter average")
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal",
               xlabel="x: shoot 3 probability", ylabel="y: guard paint probability",
               title=f"Entropy tau={tau}")
        ax.legend(fontsize=8, loc="lower left")
        ax = axes[1, column]
        ax.plot(indices, path[indices, 0], label="x", lw=0.6)
        ax.plot(indices, path[indices, 1], label="y", lw=0.6)
        for value in EQUILIBRIUM:
            ax.axhline(value, color="gray", ls=":", lw=0.8)
        ax.set(xlabel="Possession", ylabel="Action probability", ylim=(0, 1))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.legend()
        np.savetxt(out / f"2c_trajectory_tau_{tau:g}.csv",
                   np.column_stack([indices, path[indices]]), delimiter=",",
                   header="possession,x,y", comments="")
    fig.suptitle(f"400,000 stochastic possessions; eta=0.02; seed={seed}")
    fig.tight_layout()
    fig.savefig(out / "2c_learning.png", dpi=180)
    plt.close(fig)
    result = {"equilibrium": EQUILIBRIUM.tolist(), "mean_flow": flow_result,
              "baseline": "Arithmetic mean of all previous points; defense uses its negative",
              "distance": "Euclidean distance in probability space", "simulations": rows}
    (out / "2c_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("hw6_results"))
    parser.add_argument("--seed", type=int, default=4370)
    args = parser.parse_args()
    run(args.output_dir, args.seed)
