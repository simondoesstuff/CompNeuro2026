"""HW3 1(c): firing numbers of the periodically forced LIF neuron.

The forcing period is one time unit, so the spike rate is also the number
of spikes per forcing period. Subthreshold voltage is propagated exactly;
a time grid brackets threshold crossings, which are refined by bisection.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FORCING_AMPLITUDE = 0.5
TRANSIENT = 300
OBSERVATION_DURATION = 1000
PHASE_CURRENT = 1.6
THRESHOLD = 1.0
OMEGA = 2 * np.pi


def filtered_current(time, current, epsilon):
    """Periodic solution G(t) of G' = I + epsilon*sin(2*pi*t) - G."""
    return current + epsilon * (
        np.sin(OMEGA * time) - OMEGA * np.cos(OMEGA * time)
    ) / (1 + OMEGA**2)


def crossing_offsets(time, currents, coefficients, epsilon, dt):
    """Locate the first threshold crossing within each bracket [0, dt].

    Between resets, v(time + s) = G(time + s) + (v(time) - G(time))*exp(-s).
    The caller supplies the coefficient v(time) - G(time).
    """
    lower = np.zeros_like(currents)
    upper = np.full_like(currents, dt)
    for _ in range(30):
        midpoint = (lower + upper) / 2
        voltage = filtered_current(time + midpoint, currents, epsilon)
        voltage += coefficients * np.exp(-midpoint)
        above_threshold = voltage >= THRESHOLD
        upper = np.where(above_threshold, midpoint, upper)
        lower = np.where(above_threshold, lower, midpoint)
    return (lower + upper) / 2


def simulate(currents, epsilon=0.5, transient=300, duration=1000, dt=0.02):
    """Return firing numbers and spike times for the current nearest 1.6.

    Start at reset voltage zero and discard the transient when counting spikes.
    For the default current range and step size there is at most one spike
    per step. The grid must be fine enough to bracket the first crossing.
    """
    currents = np.asarray(currents, dtype=float)
    voltage = np.zeros_like(currents)
    spike_counts = np.zeros_like(currents, dtype=int)
    phase_spikes = []
    phase_index = int(np.argmin(abs(currents - PHASE_CURRENT)))
    observation_end = transient + duration
    total_steps = round(observation_end / dt)

    for step in range(total_steps):
        time = step * dt
        step_end = time + dt
        coefficients = voltage - filtered_current(time, currents, epsilon)
        endpoint_voltage = filtered_current(step_end, currents, epsilon)
        endpoint_voltage += coefficients * np.exp(-dt)
        crossing = endpoint_voltage >= THRESHOLD

        if np.any(crossing):
            crossing_currents = currents[crossing]
            offsets = crossing_offsets(
                time, crossing_currents, coefficients[crossing], epsilon, dt
            )
            spike_times = time + offsets
            observed = (spike_times >= transient) & (spike_times < observation_end)
            spike_counts[crossing] += observed

            # Reset v to zero at the spike, then propagate to the grid endpoint.
            endpoint_voltage[crossing] = filtered_current(
                step_end, crossing_currents, epsilon
            ) - filtered_current(spike_times, crossing_currents, epsilon) * np.exp(
                -(step_end - spike_times)
            )

            crossing_indices = np.flatnonzero(crossing)
            phase_match = np.flatnonzero(crossing_indices == phase_index)
            if phase_match.size:
                spike_time = spike_times[phase_match[0]]
                if spike_time >= transient:
                    phase_spikes.append(spike_time)

        voltage = endpoint_voltage

    return spike_counts / duration, np.asarray(phase_spikes)


def unforced_firing_numbers(currents):
    """Unforced LIF rate; inputs at or below threshold never fire."""
    rates = np.zeros_like(currents)
    active = currents > THRESHOLD
    rates[active] = 1 / np.log(currents[active] / (currents[active] - THRESHOLD))
    return rates


def longest_plateau(currents, rates, target):
    """Return the bounds of the longest sampled plateau, or None if absent.

    The tolerance allows one spike of counting error over 1000 periods.
    """
    indices = np.flatnonzero(abs(rates - target) <= 0.0011)
    if indices.size == 0:
        return None
    groups = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    longest = max(groups, key=len)
    return currents[longest[0]], currents[longest[-1]]


def plot_firing_numbers(currents, rates, unforced, boundaries, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(currents, rates, label=r"Forced, $\epsilon=0.5$")
    ax.plot(currents, unforced, "--", label="Unforced")
    for index, boundary in enumerate(boundaries):
        ax.axvline(
            boundary, color="black", ls=":",
            label="Candidate 1:1 boundaries" if index == 0 else None,
        )

    for numerator, denominator in [(1, 2), (2, 3), (1, 1), (3, 2), (2, 1)]:
        target = numerator / denominator
        bounds = longest_plateau(currents, rates, target)
        if bounds is None:
            continue
        left, right = bounds
        print(f"{numerator}:{denominator} plateau sampled at I = [{left:.4f}, {right:.4f}]")
        ax.annotate(
            f"{numerator}:{denominator}", ((left + right) / 2, target),
            xytext=(0, 12), textcoords="offset points", ha="center",
        )

    ax.set(
        xlabel="Mean input $I$ (dimensionless)",
        ylabel="Firing number (spikes / forcing period)",
        title="HW3 1(c): forcing period = 1; transient = 300, observation = 1000 periods",
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "1c_firing_numbers.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("hw3_results"))
    parser.add_argument("--dt", type=float, default=0.02)
    args = parser.parse_args()
    if args.dt <= 0:
        parser.error("--dt must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    currents = np.linspace(0.9, 3, 1051)
    rates, spikes = simulate(
        currents, epsilon=FORCING_AMPLITUDE, transient=TRANSIENT,
        duration=OBSERVATION_DURATION, dt=args.dt,
    )
    unforced = unforced_firing_numbers(currents)
    locking_center = 1 / (1 - np.exp(-1))
    locking_width = FORCING_AMPLITUDE / np.sqrt(1 + 4 * np.pi**2)
    boundaries = locking_center + np.array([-1, 1]) * locking_width

    np.savetxt(
        args.output_dir / "1c_firing_numbers.csv",
        np.column_stack((currents, rates, unforced)), delimiter=",",
        header="I,forced_spikes_per_period,unforced_spikes_per_period", comments="",
    )
    print(f"Candidate 1:1 boundaries: {boundaries[0]:.6f}, {boundaries[1]:.6f}")
    plot_firing_numbers(currents, rates, unforced, boundaries, args.output_dir)

    # Circular averaging respects the wrap from phase 1 back to phase 0.
    phases = spikes % 1
    mean_phase = np.angle(np.mean(np.exp(2j * np.pi * phases))) / (2 * np.pi) % 1
    maximum_deviation = np.max(abs(phases - mean_phase))
    print(
        f"I=1.6: spike phase = {mean_phase:.8f}, "
        f"maximum phase deviation = {maximum_deviation:.3g}"
    )


if __name__ == "__main__":
    main()
