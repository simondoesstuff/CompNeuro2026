"""HW4 1(c): ocular-dominance stripe formation on a periodic ring."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

N = 200
DT = 0.02
END_TIME = 40.0
SIGMA_E = 1.0
SIGMA_I_VALUES = (2.0, 3.0)
N_CONDITIONS = 20


def ring_kernel(sigma_i, n=N):
    """Mexican-hat kernel sampled at shortest periodic distances."""
    distance = np.minimum(np.arange(n), n - np.arange(n))
    return np.exp(-distance**2 / (2 * SIGMA_E**2)) - (
        SIGMA_E / sigma_i
    ) * np.exp(-distance**2 / (2 * sigma_i**2))


def fastest_ring_mode(kernel):
    """Return the fastest positive nonuniform DFT mode and its period."""
    n = len(kernel)
    growth = np.fft.fft(kernel).real
    growth[0] = -np.inf
    mode = int(np.argmax(growth))
    if growth[mode] <= 0:
        raise RuntimeError("No growing nonuniform Fourier mode")
    mode = min(mode, n - mode)
    return mode, n / mode, growth


def stripe_period(weights):
    """Estimate one stripe-pair period from sign runs around the ring."""
    signs = np.where(weights >= 0, 1, -1)
    transitions = np.count_nonzero(signs != np.roll(signs, 1))
    pairs = max(1, transitions // 2)
    return len(weights) / pairs, pairs


def simulate(sigma_i, rng, *, n=N, dt=DT, end_time=END_TIME):
    kernel = ring_kernel(sigma_i, n)
    kernel_hat = np.fft.fft(kernel)
    weights = rng.normal(0.0, 0.01, n)
    for _ in range(round(end_time / dt)):
        weights += dt * np.fft.ifft(kernel_hat * np.fft.fft(weights)).real
        np.clip(weights, -1.0, 1.0, out=weights)
    return weights


def run(output_dir, seed=4370):
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(SIGMA_I_VALUES), 1, figsize=(10, 6), sharex=True)
    summaries = []
    for ax, sigma_i in zip(axes, SIGMA_I_VALUES):
        kernel = ring_kernel(sigma_i)
        mode, ring_period, _ = fastest_ring_mode(kernel)
        continuous_k = np.sqrt(
            4 * np.log(sigma_i / SIGMA_E) / (sigma_i**2 - SIGMA_E**2)
        )
        continuous_period = 2 * np.pi / continuous_k
        periods, pairs = [], []
        example = None
        for _ in range(N_CONDITIONS):
            example = simulate(sigma_i, rng)
            period, pair_count = stripe_period(example)
            periods.append(period)
            pairs.append(pair_count)
        ax.plot(np.arange(N), example, lw=1.4)
        ax.axhline(0, color="black", lw=0.7)
        ax.set(ylabel="$w_-(a)$", title=fr"$\sigma_i={sigma_i:g}$, final pattern")
        summaries.append((sigma_i, mode, continuous_period, ring_period,
                          float(np.mean(periods)), float(np.std(periods, ddof=1)),
                          float(np.mean(pairs))))
        np.savetxt(
            output_dir / f"1c_periods_sigma_i_{sigma_i:g}.csv",
            np.column_stack((periods, pairs)), delimiter=",",
            header="measured_period_sites,stripe_pairs", comments="",
        )
    axes[-1].set(xlabel="Ring position $a$ (sites)", xlim=(0, N - 1))
    fig.tight_layout()
    fig.savefig(output_dir / "1c_ocular_dominance.png", dpi=180)
    plt.close(fig)
    with (output_dir / "1c_summary.csv").open("w") as stream:
        stream.write("sigma_i,fastest_mode,continuous_period,ring_mode_period,"
                     "mean_measured_period,sd_measured_period,mean_stripe_pairs\n")
        for row in summaries:
            stream.write(",".join(f"{value:.8g}" for value in row) + "\n")
    for row in summaries:
        print("sigma_i={:.1f}: k_mode={}, continuous period={:.4f}, "
              "ring period={:.4f}, measured={:.4f} +/- {:.4f} sites, "
              "mean pairs={:.2f}".format(*row))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("hw4_results"))
    parser.add_argument("--seed", type=int, default=4370)
    args = parser.parse_args()
    run(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
