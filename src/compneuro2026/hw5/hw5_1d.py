"""HW5 1(d): Wilson-Cowan limit cycles, frequency and peak lag."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit


def simulate(tau_i, epsilon, rtol=1e-9):
    w = (1 + epsilon) * (1 + 5 / tau_i) / 0.36
    p = np.log(1 / 9) / 4

    def rhs(t, y):
        e, i = y
        return [
            (-e + expit(4 * (w * e - 10 * i + p - 0.1 * w + 1))) / 5,
            (-i + expit(4 * (10 * e + p - 1))) / tau_i,
        ]

    def peak_e(t, y):
        return rhs(t, y)[0]

    def peak_i(t, y):
        return rhs(t, y)[1]

    def trough(t, y):
        return rhs(t, y)[0]

    peak_e.direction = peak_i.direction = -1
    trough.direction = 1
    state = [0.101, 0.1]
    previous = None
    # Compare peak and trough levels over consecutive 1000-ms blocks.
    for block in range(100):
        sol = solve_ivp(
            rhs,
            (0, 1000),
            state,
            method="DOP853",
            rtol=rtol,
            atol=rtol * 0.01,
            max_step=1,
            events=(peak_e, trough),
        )
        state = sol.y[:, -1]
        if not sol.success:
            raise RuntimeError(sol.message)
        levels = np.array(
            [np.mean(sol.y_events[0][-5:, 0]), np.mean(sol.y_events[1][-5:, 0])]
        )
        if previous is not None and np.max(abs(levels - previous)) < 1e-7:
            break
        previous = levels
    else:
        raise RuntimeError("Limit cycle did not settle")
    sol = solve_ivp(
        rhs,
        (0, 1000),
        state,
        method="DOP853",
        rtol=rtol,
        atol=rtol * 0.01,
        max_step=0.5,
        events=(peak_e, peak_i),
        dense_output=True,
    )
    te, ti = sol.t_events
    period = float(np.mean(np.diff(te)))
    delays = [ti[ti > t][0] - t for t in te if np.any(ti > t)]
    # Troughs are located with the opposite event direction.
    trough_sol = solve_ivp(
        rhs,
        (0, 2 * period),
        sol.y[:, -1],
        method="DOP853",
        rtol=rtol,
        atol=rtol * 0.01,
        max_step=0.1,
        events=trough,
    )
    amplitude = float(
        np.mean(sol.y_events[0][-10:, 0]) - np.mean(trough_sol.y_events[0][:, 0])
    )
    omega = np.sqrt(2.592 / tau_i - 1 / tau_i**2)
    row = {
        "tau_i": tau_i,
        "epsilon": epsilon,
        "period": period,
        "amplitude": amplitude,
        "lag_deg": 360 * float(np.mean(delays)) / period,
        "frequency": 1000 / period,
        "linear_frequency": 1000 * omega / (2 * np.pi),
        "settling_ms": (block + 1) * 1000,
    }
    times = np.linspace(0, 6 * period, 2401)
    return row, times, sol.sol(times)


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    rows = [simulate(10, eps)[0] for eps in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]]
    frequency_rows = [simulate(tau, 0.02)[0] for tau in [2, 3, 5, 7.5, 10, 15, 20]]
    x = np.array([r["epsilon"] for r in rows])
    y = np.array([r["amplitude"] for r in rows])
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    near_slope = float(np.polyfit(np.log(x[:3]), np.log(y[:3]), 1)[0])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].loglog(x, y, "o", label="Simulated peak-to-trough")
    axes[0].loglog(
        x, np.exp(intercept) * x**slope, "--", label=f"Fit slope = {slope:.4f}"
    )
    axes[0].set(
        xlabel=r"$\epsilon$", ylabel=r"$\max E-\min E$", title=r"$\tau_I=10$ ms"
    )
    axes[0].legend()
    axes[1].plot(
        [r["tau_i"] for r in frequency_rows],
        [r["frequency"] for r in frequency_rows],
        "o-",
        label="Simulation, epsilon=0.02",
    )
    axes[1].plot(
        [r["tau_i"] for r in frequency_rows],
        [r["linear_frequency"] for r in frequency_rows],
        "--",
        label="Trace-zero linear theory",
    )
    axes[1].set(xlabel=r"$\tau_I$ (ms)", ylabel="Frequency (Hz)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "1d_scaling_frequency.png", dpi=180)
    plt.close(fig)
    _row, t, y = simulate(10, 0.1)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(t, y[0], label="E")
    ax.plot(t, y[1], label="I")
    ax.set(
        xlabel="Time after settling (ms)",
        ylabel="Population rate",
        title=r"$\tau_I=10$ ms, $\epsilon=0.1$",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "1d_time_series.png", dpi=180)
    plt.close(fig)
    result = {
        "epsilon_sweep": rows,
        "frequency_sweep": frequency_rows,
        "slope": float(slope),
        "near_slope": near_slope,
    }
    (out / "1d_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("hw5_results"))
    run(parser.parse_args().output_dir)
