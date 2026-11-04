"""HW5 2(d): stiff rivalry dynamics and dominance hand-overs."""

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit


def simulate(il, w=0.8, rtol=1e-8, max_step=1):
    def rhs(t, y):
        l, r, a, b = y
        return [
            -l + expit(1000 * (w * l - r - 2 * a + il)),
            -r + expit(1000 * (w * r - l - 2 * b + 1.3)),
            (-a + l) / 120,
            (-b + r) / 120,
        ]

    def left(t, y):
        return y[0] - y[1] - 0.5

    def right(t, y):
        return y[1] - y[0] - 0.5

    left.direction = right.direction = 1
    sol = solve_ivp(
        rhs,
        (0, 3600),
        [1, 0, 0, 0],
        method="Radau",
        rtol=rtol,
        atol=rtol * 0.01,
        max_step=max_step,
        events=(left, right),
        dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    # A hand-over occurs when the incoming percept first meets its threshold.
    # Do not count the initial or terminal censored interval; discard three full intervals.
    entries = sorted(
        [(t, "L") for t in sol.t_events[0]] + [(t, "R") for t in sol.t_events[1]]
    )
    intervals = [
        (label, end - start)
        for (start, label), (end, next_label) in itertools.pairwise(entries)
        if label != next_label
    ][3:]
    durations = {
        label: [duration for owner, duration in intervals if owner == label]
        for label in ["L", "R"]
    }
    bl, br = (il - 1) / 2, 0.15
    theory = (
        [120 * np.log((1 - bl) / br), 120 * np.log((1 - br) / bl)]
        if w == 0.8
        else [120 * np.log((w + 1.3) / (2 - w - 1.3))] * 2
    )
    return {
        "il": il,
        "w": w,
        "mean_l": float(np.mean(durations["L"])),
        "mean_r": float(np.mean(durations["R"])),
        "count_l": len(durations["L"]),
        "count_r": len(durations["R"]),
        "theory_l": float(theory[0]),
        "theory_r": float(theory[1]),
    }, sol


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for il, w in [(1.3, 0.8), (1.45, 0.8), (1.6, 0.8), (1.3, 0.3)]:
        row, sol = simulate(il, w)
        rows.append(row)
        if il == 1.3 and w == 0.8:
            t = np.linspace(0, 3600, 36001)
            y = sol.sol(t)
            fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
            for j, name in enumerate(["L", "R"]):
                axes[0].plot(t, y[j], label=f"$u_{name}$")
                axes[1].plot(t, y[j + 2], label=f"$a_{name}$")
            for level, name in [(0.15, "B=0.15"), (0.85, "1-B=0.85")]:
                axes[1].axhline(level, color="gray", ls="--", label=name)
            axes[0].set(
                ylabel="Rate",
                title=r"$I_L=I_R=1.3$, $w=0.8$, $\eta=1000$, $\tau_a=120$",
            )
            axes[1].set(xlabel="Time (rate time constants)", ylabel="Adaptation")
            for ax in axes:
                ax.legend(loc="upper right")
            fig.tight_layout()
            fig.savefig(out / "2d_rivalry.png", dpi=180)
            plt.close(fig)
    (out / "2d_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("hw5_results"))
    run(parser.parse_args().output_dir)
