"""HW5 3(d): zero-padded Euler neural field and bump relaxation."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.special import expit


def primitive(x):
    return 2 * np.sign(x) * (np.exp(-abs(x) / 2) - np.exp(-abs(x)))


def widths(kappa):
    s = np.sqrt(1 - 2 * kappa)
    return -np.log((1 + s) / 2), -np.log((1 - s) / 2)


def eigenvalue(a):
    w = 2 * np.exp(-2 * a) - np.exp(-a)
    return 2 * w / (1 - w)


def half_width(x, u, kappa):
    indices = np.flatnonzero(u >= kappa)
    if not len(indices):
        return 0.0
    if np.any(np.diff(indices) != 1):
        raise RuntimeError("Active set is no longer one interval")
    l, r = indices[0], indices[-1]
    if l == 0 or r == len(x) - 1:
        raise RuntimeError("Bump reached boundary")
    left = x[l - 1] + (kappa - u[l - 1]) * (x[l] - x[l - 1]) / (u[l] - u[l - 1])
    right = x[r] + (kappa - u[r]) * (x[r + 1] - x[r]) / (u[r + 1] - u[r])
    return (right - left) / 2


def simulate(kappa, a0, end=160, dt=0.02, dx=0.02):
    x = np.linspace(-20, 20, round(40 / dx) + 1)
    n = len(x)
    # Full lag kernel: crop the *linear* convolution at index n-1.
    lag = np.arange(-(n - 1), n) * dx
    kernel = 2 * np.exp(-abs(lag)) - np.exp(-abs(lag) / 2)
    size = next_fast_len(n + len(kernel) - 1)
    kernel_hat = rfft(kernel, size)

    def convolution(f):
        return dx * irfft(kernel_hat * rfft(f, size), size)[n - 1 : 2 * n - 1]

    u = primitive(x + a0) - primitive(x - a0)
    time = np.arange(round(end / dt) + 1) * dt
    a = np.empty(len(time))
    a[0] = half_width(x, u, kappa)
    for step in range(1, len(time)):
        u += dt * (-u + convolution(expit(100 * (u - kappa))))
        a[step] = half_width(x, u, kappa)
    return time, a, x, u, convolution


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    narrow, wide = widths(0.3)
    fig, ax = plt.subplots(figsize=(9, 4))
    outcomes = []
    for name, a0 in [
        ("narrow - 0.02", narrow - 0.02),
        ("narrow + 0.02", narrow + 0.02),
        ("wide - 0.05", wide - 0.05),
        ("wide + 0.05", wide + 0.05),
    ]:
        t, a, *_ = simulate(0.3, a0)
        ax.plot(t, a, label=name)
        outcomes.append(
            {
                "initial": name,
                "a0": float(a0),
                "initial_active_half_width": float(a[0]),
                "final_half_width": float(a[-1]),
            }
        )
    ax.axhline(narrow, color="gray", ls=":", label="Heaviside narrow width")
    ax.axhline(wide, color="black", ls="--", label="Heaviside wide width")
    ax.set(
        xlabel="Time (field time units)",
        ylabel="Active half-width (space units)",
        title=r"$\kappa=0.3$, $\Delta x=\Delta t=0.02$, sigmoid slope 100",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "3d_bump_widths.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    rows = []
    for ax, kappa in zip(axes, [0.1, 0.3, 0.45]):
        _, wide = widths(kappa)
        t, a, _x, u, conv = simulate(kappa, wide + 0.05)
        # Use an early window to report the requested distance to the analytical
        # Heaviside width; smoothing leaves a nonzero late-time offset.
        raw = abs(a - wide)
        mask = (t >= 1) & (t <= 8)
        raw_rate, raw_intercept = np.polyfit(t[mask], np.log(raw[mask]), 1)
        equilibrium = a[-1]
        corrected = abs(a - equilibrium)
        fit_mask = (t >= 1) & (t <= 15) & (corrected > 1e-5)
        corrected_rate, corrected_intercept = np.polyfit(
            t[fit_mask], np.log(corrected[fit_mask]), 1
        )
        theory = eigenvalue(wide)
        euler_rate = np.log1p(0.02 * theory) / 0.02
        ax.semilogy(t, raw, label=r"$|a-a_+|$")
        ax.semilogy(t, corrected, label=r"$|a-a_\infty|$")
        ax.semilogy(
            t[mask],
            np.exp(raw_intercept + raw_rate * t[mask]),
            "k--",
            label="Raw fit (1 to 8)",
        )
        ax.semilogy(
            t[fit_mask],
            np.exp(corrected_intercept + corrected_rate * t[fit_mask]),
            ":",
            color="red",
            label="Shift-corrected fit",
        )
        ax.set(
            xlim=(0, 50),
            ylim=(1e-6, 0.1),
            xlabel="Time (field time units)",
            ylabel="Half-width error",
            title=rf"$\kappa={kappa}$",
        )
        ax.legend(fontsize=7)
        rows.append(
            {
                "kappa": kappa,
                "heaviside_width": float(wide),
                "sigmoid_width": float(equilibrium),
                "raw_fit_rate": float(raw_rate),
                "corrected_fit_rate": float(corrected_rate),
                "theory_rate": float(theory),
                "euler_theory_rate": float(euler_rate),
                "final_rhs_max": float(
                    np.max(abs(-u + conv(expit(100 * (u - kappa)))))
                ),
            }
        )
        np.savetxt(
            out / f"3d_relaxation_kappa_{kappa:g}.csv",
            np.column_stack([t, a]),
            delimiter=",",
            header="time,half_width",
            comments="",
        )
    fig.tight_layout()
    fig.savefig(out / "3d_relaxation.png", dpi=180)
    plt.close(fig)
    result = {"outcomes": outcomes, "relaxation": rows}
    (out / "3d_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("hw5_results"))
    run(parser.parse_args().output_dir)
