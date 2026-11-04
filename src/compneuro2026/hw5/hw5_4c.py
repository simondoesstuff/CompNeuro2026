"""HW5 optional 4(c): periodic Turing patterns with exact Fourier kernel."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def simulate(gamma, noise, dt=0.02):
    n = len(noise)
    k = 2 * np.pi * np.fft.rfftfreq(n, d=60 / n)
    kernel_hat = 12 * k * k / ((1 + k * k) * (1 + 4 * k * k))

    def rhs(u):
        return -u + np.fft.irfft(kernel_hat * np.fft.rfft(np.tanh(gamma * u)), n=n)

    u = noise.copy()
    # RK4, with the continuous Fourier multiplier (no extra dx factor).
    for _ in range(round(200 / dt)):
        a = rhs(u)
        b = rhs(u + dt * a / 2)
        c = rhs(u + dt * b / 2)
        d = rhs(u + dt * c)
        u += dt * (a + 2 * b + 2 * c + d) / 6
    spectrum = abs(np.fft.rfft(u))
    spectrum[0] = 0
    mode = int(np.argmax(spectrum))
    return u, {
        "gamma": gamma,
        "mode": mode,
        "wavenumber": float(k[mode]),
        "amplitude": float(np.ptp(u)),
    }


def run(out, seed=4370):
    out.mkdir(parents=True, exist_ok=True)
    noise = np.random.default_rng(seed).normal(0, 1e-3, 600)
    x = np.arange(600) * 0.1
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    rows = []
    for ax, gamma in zip(axes, [0.7, 1, 1.2, 2]):
        u, row = simulate(gamma, noise)
        rows.append(row)
        ax.plot(x, u)
        ax.set(
            ylabel="u",
            title=rf"$\gamma={gamma}$: n={row['mode']}, k={row['wavenumber']:.4f}, amplitude={row['amplitude']:.4g}",
        )
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        np.savetxt(
            out / f"4c_final_gamma_{gamma:g}.csv",
            np.column_stack([x, u]),
            delimiter=",",
            header="x,u",
            comments="",
        )
    axes[-1].set(xlabel="Position (space units)")
    fig.suptitle(f"Periodic field: L=60, N=600, t=200, noise SD=0.001, seed={seed}")
    fig.tight_layout()
    fig.savefig(out / "4c_turing_patterns.png", dpi=180)
    plt.close(fig)
    result = {"seed": seed, "rows": rows}
    (out / "4c_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("hw5_results"))
    parser.add_argument("--seed", type=int, default=4370)
    args = parser.parse_args()
    run(args.output_dir, args.seed)
