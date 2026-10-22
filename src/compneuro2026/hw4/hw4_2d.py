"""HW4 2(d): Euler-Maruyama decision simulations for a two-population circuit."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, solve_bvp

MU, SIGMA, I0 = 0.2, 0.5, 5.0
LAMBDAS = (0.0, 0.2)
THRESHOLDS = np.arange(0.5, 4.01, 0.5)
EXTRA_PLOT_THRESHOLDS = (4.5, 5.0)
DT, N_TRIALS = 0.002, 10_000


def theory_accuracy(theta, lam):
    if lam == 0:
        return 1 / (1 + np.exp(-MU * theta / SIGMA**2))
    g = lambda x: np.exp(-(lam * x**2 / 2 + MU * x) / SIGMA**2)
    return quad(g, -theta, 0, epsabs=1e-11)[0] / quad(
        g, -theta, theta, epsabs=1e-11
    )[0]


def theory_time(theta, lam):
    """Solve the mean first-passage backward ODE with zero endpoint values."""
    x = np.linspace(-theta, theta, 250)
    initial = np.vstack((
        np.maximum(0, theta**2 - x**2) / (2 * SIGMA**2),
        -x / SIGMA**2,
    ))

    def ode(x, y):
        drift = lam * x + MU
        return np.vstack((y[1], (-1 - drift * y[1]) / SIGMA**2))

    def boundary(ya, yb):
        return np.array([ya[0], yb[0]])

    solution = solve_bvp(ode, boundary, x, initial, tol=1e-7, max_nodes=5000)
    if not solution.success:
        raise RuntimeError(solution.message)
    return float(solution.sol(0.0)[0])


def simulate(theta, lam, rng, n_trials=N_TRIALS, dt=DT, batch_size=2000):
    """Simulate threshold crossings in batches, preserving independent noise."""
    decisions, decision_times = [], []
    diffusion_step = np.sqrt(2) * SIGMA * np.sqrt(dt)
    drift_lambda = lam
    for start in range(0, n_trials, batch_size):
        size = min(batch_size, n_trials - start)
        d = np.zeros(size)
        times = np.zeros(size)
        active = np.ones(size, dtype=bool)
        choice = np.zeros(size, dtype=np.int8)
        while np.any(active):
            idx = np.flatnonzero(active)
            d[idx] += (drift_lambda * d[idx] + MU) * dt + diffusion_step * rng.standard_normal(len(idx))
            times[idx] += dt
            crossed = np.abs(d[idx]) >= theta
            if np.any(crossed):
                finished = idx[crossed]
                choice[finished] = (d[finished] > 0).astype(np.int8)
                active[finished] = False
        decisions.append(choice)
        decision_times.append(times)
    choices = np.concatenate(decisions)
    times = np.concatenate(decision_times)
    return float(np.mean(choices)), float(np.mean(times)), choices, times


def run(output_dir, seed=4370, n_trials=N_TRIALS):
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    # Run the original conditions first to preserve their seeded results.
    conditions = [
        (lam, theta)
        for thresholds in (THRESHOLDS, EXTRA_PLOT_THRESHOLDS)
        for lam in LAMBDAS
        for theta in thresholds
    ]
    for lam, theta in conditions:
        accuracy, mean_time, choices, times = simulate(theta, lam, rng, n_trials)
        pred_accuracy = theory_accuracy(theta, lam)
        pred_time = theory_time(theta, lam)
        rows.append((lam, theta, accuracy, mean_time, pred_accuracy, pred_time,
                     np.std(choices, ddof=1) / np.sqrt(n_trials),
                     np.std(times, ddof=1) / np.sqrt(n_trials)))
        print(f"lambda={lam:.1f}, theta={theta:.1f}: accuracy={accuracy:.4f} "
              f"(theory {pred_accuracy:.4f}), mean time={mean_time:.4f} "
              f"(theory {pred_time:.4f})")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for lam in LAMBDAS:
        subset = sorted((row for row in rows if row[0] == lam), key=lambda row: row[1])
        ax = axes[0]
        ax.plot([r[1] for r in subset], [r[4] for r in subset], label=fr"Theory $\lambda={lam:g}$")
        ax.errorbar([r[1] for r in subset], [r[2] for r in subset],
                    yerr=[1.96 * r[6] for r in subset], fmt="o", ms=4,
                    label=fr"Simulation $\lambda={lam:g}$")
        ax = axes[1]
        ax.plot([r[1] for r in subset], [r[5] for r in subset], label=fr"Theory $\lambda={lam:g}$")
        ax.errorbar([r[1] for r in subset], [r[3] for r in subset],
                    yerr=[1.96 * r[7] for r in subset], fmt="o", ms=4,
                    label=fr"Simulation $\lambda={lam:g}$")
    axes[0].set(xlabel="Threshold $\\theta$", ylabel="Accuracy", ylim=(0.45, 1.01))
    axes[1].set(xlabel="Threshold $\\theta$", ylabel="Mean decision time")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "2d_decision_performance.png", dpi=180)
    plt.close(fig)
    np.savetxt(
        output_dir / "2d_summary.csv", np.asarray(rows), delimiter=",",
        header="lambda,theta,sim_accuracy,sim_mean_time,theory_accuracy,theory_mean_time,"
               "accuracy_sem,time_sem", comments="",
    )
    # Match lambda=0.2, theta=4 accuracy to the perfect-integrator prediction.
    target = next(r[2] for r in rows if r[0] == 0.2 and r[1] == 4.0)
    theta_match = SIGMA**2 / MU * np.log(target / (1 - target))
    matched_time = theory_time(theta_match, 0.0)
    observed_time = next(r[3] for r in rows if r[0] == 0.2 and r[1] == 4.0)
    print(f"lambda=0.2, theta=4: simulated accuracy={target:.4f}, "
          f"time={observed_time:.4f}; lambda=0 matched threshold={theta_match:.4f}, "
          f"predicted time={matched_time:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("hw4_results"))
    parser.add_argument("--seed", type=int, default=4370)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    args = parser.parse_args()
    run(args.output_dir, args.seed, args.trials)


if __name__ == "__main__":
    main()
