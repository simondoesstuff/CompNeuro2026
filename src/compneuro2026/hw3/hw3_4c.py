"""HW3 4(c): synchronization of gap-junction coupled QIF neurons.

The voltage simulation uses finite thresholds +/-200 in place of infinity.
Spike lags are compared with the averaged phase model, whose prediction is
tan(pi*phi(t)) = tan(pi*phi(0))*exp(-2*epsilon*t).
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

VOLTAGE_CUTOFF = 200
INITIAL_PHASE_LAG = 0.4
COUPLING_STRENGTHS = (0.005, 0.02, 0.05)


@dataclass
class SynchronizationResult:
    """Measured lags, analytical prediction, and fitted decay for one coupling."""

    epsilon: float
    times: np.ndarray
    lags: np.ndarray
    prediction: np.ndarray
    rate: float
    intercept: float
    fit_mask: np.ndarray


def spike_lags(spikes):
    """Pair a neuron-1 spike with the next neuron-2 spike in the same cycle.

    Normalize the time difference by neuron 1's measured interspike interval.
    Return each cycle's start time and the corresponding lag in cycles.
    """
    first_spikes, second_spikes = map(np.asarray, spikes)
    times, lags = [], []
    for cycle_start, cycle_end in zip(first_spikes[:-1], first_spikes[1:]):
        following_index = np.searchsorted(second_spikes, cycle_start, side="right")
        if following_index >= len(second_spikes):
            continue
        following_spike = second_spikes[following_index]
        if following_spike < cycle_end:
            times.append(cycle_start)
            lags.append((following_spike - cycle_start) / (cycle_end - cycle_start))
    return np.asarray(times), np.asarray(lags)


def simulate(epsilon, cutoff=200, phi0=0.4):
    """Integrate until 3/epsilon, resetting each neuron when it reaches cutoff.

    For an uncoupled finite-cutoff QIF neuron, the angle arctan(v) advances
    uniformly from -arctan(cutoff) to +arctan(cutoff) each cycle. Neuron 1
    starts at reset; neuron 2 starts phi0 cycles behind it (phase 1-phi0).
    """
    half_angle = np.arctan(cutoff)
    period = 2 * half_angle
    second_voltage = np.tan(-half_angle + (1 - phi0) * period)
    state = np.array([-cutoff, second_voltage])
    spikes = [[0.0], []]
    end_time = 3 / epsilon
    time = 0.0

    def voltage_derivatives(time, voltages):
        first_voltage, second_voltage = voltages
        gap_current = epsilon * (second_voltage - first_voltage)
        return [
            1 + first_voltage**2 + gap_current,
            1 + second_voltage**2 - gap_current,
        ]

    def first_threshold(time, voltages):
        return voltages[0] - cutoff

    def second_threshold(time, voltages):
        return voltages[1] - cutoff

    # Stop at a spike so the reset occurs before integration resumes.
    first_threshold.terminal = second_threshold.terminal = True
    first_threshold.direction = second_threshold.direction = 1
    while time < end_time:
        solution = solve_ivp(
            voltage_derivatives, (time, end_time), state,
            events=(first_threshold, second_threshold),
            rtol=1e-10, atol=1e-11, max_step=0.1, method="DOP853",
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        time = solution.t[-1]
        state = solution.y[:, -1].copy()
        fired = False
        for neuron in range(2):
            if solution.t_events[neuron].size:
                spikes[neuron].append(time)
                state[neuron] = -cutoff
                fired = True
        if not fired:
            break
    return spike_lags(spikes)


def transformed_lags(lags):
    """The averaged model predicts log(tan(pi*lag)) is linear in time."""
    return np.log(np.tan(np.pi * lags))


def decay_fit(times, lags):
    """Fit the exponential decay rate after discarding early cycles.

    Keep lags above numerical resolution and away from the tangent's pole
    at half a cycle. The slope should be -2*epsilon in the averaged model.
    """
    fit_mask = (times > 5 * np.pi) & (lags > 1e-4) & (lags < 0.45)
    if np.count_nonzero(fit_mask) < 2:
        raise ValueError("At least two resolved lags are needed to fit the decay")
    slope, intercept = np.polyfit(times[fit_mask], transformed_lags(lags[fit_mask]), 1)
    return -slope, intercept, fit_mask


def averaged_lags(times, epsilon, initial_lag=INITIAL_PHASE_LAG):
    """Analytical phase-model prediction for an initial lag below half a cycle."""
    decaying_tangent = np.tan(np.pi * initial_lag) * np.exp(-2 * epsilon * times)
    return np.arctan(decaying_tangent) / np.pi


def plot_synchronization(experiments, output_dir):
    fig, (lag_ax, decay_ax) = plt.subplots(1, 2, figsize=(12, 5))
    for result in experiments:
        measured_line, = decay_ax.plot(
            result.times, transformed_lags(result.lags),
            label=f"Measured, $\\epsilon={result.epsilon:g}$",
        )
        fit_times = result.times[result.fit_mask]
        decay_ax.plot(
            fit_times, result.intercept - result.rate * fit_times, "--",
            color=measured_line.get_color(),
        )
        if result.epsilon == 0.02:
            lag_ax.plot(result.times, result.lags, ".", label="Voltage simulation, $L=200$")
            lag_ax.plot(result.times, result.prediction, label="Averaged prediction")

    lag_ax.set(
        xlabel="Time $t$ (model time units)", ylabel=r"Spike lag $\phi_k$ (cycles)",
        title=r"$\epsilon=0.02$, $\phi_0=0.4$, symmetric cutoff $\pm200$",
    )
    decay_ax.set(
        xlabel="Time $t$ (model time units)",
        ylabel=r"$\log[\tan(\pi\phi_k)]$ (dimensionless)",
        title="Decay rates: voltage simulation and linear fits",
    )
    for ax in (lag_ax, decay_ax):
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "4c_synchronization.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("hw3_results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    experiments = []
    for epsilon in COUPLING_STRENGTHS:
        times, lags = simulate(epsilon, VOLTAGE_CUTOFF, INITIAL_PHASE_LAG)
        rate, intercept, fit_mask = decay_fit(times, lags)
        prediction = averaged_lags(times, epsilon)
        predicted_rate = 2 * epsilon
        relative_difference = rate / predicted_rate - 1
        print(
            f"epsilon={epsilon:.4g}: fitted rate={rate:.8f}, prediction={predicted_rate:.4g}, "
            f"relative difference={relative_difference:.4%}, fit points={sum(fit_mask)}"
        )
        print(f"  initial measured lag={lags[0]:.8f}; final lag={lags[-1]:.8f}")
        np.savetxt(
            args.output_dir / f"4c_lags_eps_{epsilon:g}.csv",
            np.column_stack((times, lags, prediction)), delimiter=",",
            header="time,measured_lag_cycles,averaged_prediction_cycles", comments="",
        )
        experiments.append(SynchronizationResult(
            epsilon=epsilon, times=times, lags=lags, prediction=prediction,
            rate=rate, intercept=intercept, fit_mask=fit_mask,
        ))
    plot_synchronization(experiments, args.output_dir)


if __name__ == "__main__":
    main()
