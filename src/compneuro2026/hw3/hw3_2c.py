"""HW3 2(c): voltage-kick phase response curves for LIF and FitzHugh–Nagumo.

A positive response means the voltage kick advances the oscillator. Responses
are measured in cycles per voltage unit: (unperturbed time - kicked time)
/ (period * kick size).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

LIF_CURRENT = 1.5
LIF_THRESHOLD = 1.0
LIF_PERIOD = np.log(3)
KICK_SIZE = 1e-3
RELAXATION_CYCLES = 4


def fhn(time, state):
    """FitzHugh–Nagumo dynamics with I=0.5, a=0.7, b=0.8, eta=0.08."""
    voltage, recovery = state
    voltage_derivative = voltage - voltage**3 / 3 - recovery + 0.5
    recovery_derivative = 0.08 * (voltage + 0.7 - 0.8 * recovery)
    return [voltage_derivative, recovery_derivative]


def crossing(time, state):
    """Mark each cycle at an upward crossing of voltage zero."""
    return state[0]


crossing.direction = 1


def integrate(state, end, *, dense_output=False):
    """Integrate FHN from time zero, recording upward voltage crossings."""
    solution = solve_ivp(
        fhn, (0, end), state, events=crossing, dense_output=dense_output,
        rtol=1e-10, atol=1e-12, max_step=0.3, method="DOP853",
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution


def settled_fhn_orbit():
    """Discard the transient, then sample one cycle starting at voltage zero."""
    settled = integrate([-1, 1], 1000)
    crossing_times = settled.t_events[0]
    period = np.diff(crossing_times)[-1]
    initial_state = settled.y_events[0][-1].copy()
    initial_state[0] = 0
    orbit = integrate(initial_state, period, dense_output=True)
    return period, orbit


def fhn_prc(phases, delta=1e-3, cycles=4):
    """Kick the voltage at each phase and measure the shift after several cycles.

    Waiting for the fourth subsequent crossing lets the perturbed trajectory
    relax toward the limit cycle before measuring its lasting phase shift.
    """
    period, orbit = settled_fhn_orbit()
    responses = []
    for phase in phases:
        kick_time = phase * period
        kicked_state = orbit.sol(kick_time)
        kicked_state[0] += delta
        perturbed = integrate(kicked_state, (cycles + 1) * period)
        crossing_times = perturbed.t_events[0]
        if len(crossing_times) < cycles:
            raise RuntimeError("Insufficient post-kick crossings")

        # Crossing times are relative to the kick; compare in orbit time.
        kicked_crossing_time = kick_time + crossing_times[cycles - 1]
        unperturbed_crossing_time = cycles * period
        response = (unperturbed_crossing_time - kicked_crossing_time) / (period * delta)
        responses.append(response)
    return period, np.asarray(responses)


def lif_prc(phases, delta=KICK_SIZE):
    """Return the finite-kick LIF response and its infinitesimal analytical PRC.

    For tau=1 and reset=0, v(t)=I*(1-exp(-t)). After a kick, the remaining
    time to threshold is log((I-v-delta)/(I-threshold)), clipped to zero
    when the kick itself reaches threshold.
    """
    kick_times = phases * LIF_PERIOD
    voltages = LIF_CURRENT * (1 - np.exp(-kick_times))
    remaining_times = np.maximum(
        0, np.log((LIF_CURRENT - voltages - delta) / (LIF_CURRENT - LIF_THRESHOLD))
    )
    next_spike_times = kick_times + remaining_times
    kick_response = (LIF_PERIOD - next_spike_times) / (LIF_PERIOD * delta)
    analytical_response = np.exp(kick_times) / (LIF_CURRENT * LIF_PERIOD)
    return kick_response, analytical_response


def zero_crossing_phases(phases, responses):
    """Estimate sign changes between neighboring samples by linear interpolation."""
    indices = np.flatnonzero(np.diff(np.signbit(responses)))
    zeros = []
    for index in indices:
        phase_step = phases[index + 1] - phases[index]
        response_step = responses[index + 1] - responses[index]
        zeros.append(phases[index] - responses[index] * phase_step / response_step)
    return zeros


def report_prcs(phases, period_fhn, lif_kick, lif_formula, fhn_kick, fhn_half_kick):
    for name, period, response in [
        ("LIF", LIF_PERIOD, lif_kick), ("FHN", period_fhn, fhn_kick),
    ]:
        minimum_index = np.argmin(response)
        maximum_index = np.argmax(response)
        print(
            f"{name}: period={period:.8f}; "
            f"minimum R={response[minimum_index]:.8f} at phase={phases[minimum_index]:.6f}; "
            f"maximum R={response[maximum_index]:.8f} at phase={phases[maximum_index]:.6f}"
        )
    print(f"FHN zero crossings (linear interpolation): {zero_crossing_phases(phases, fhn_kick)}")
    print(f"FHN maximum change on halving kick: {np.max(abs(fhn_kick - fhn_half_kick)):.6g}")
    print(f"LIF maximum absolute error against formula: {np.max(abs(lif_kick - lif_formula)):.6g}")


def plot_prcs(phases, lif_kick, lif_formula, fhn_kick, fhn_half_kick, output_dir):
    fig, (lif_ax, fhn_ax) = plt.subplots(1, 2, figsize=(12, 5))
    lif_ax.plot(phases, lif_formula, label="Analytical PRC")
    lif_ax.plot(phases[::5], lif_kick[::5], ".", label=r"Voltage kick $\delta=10^{-3}$")
    lif_ax.set_title(r"LIF: $\tau=1$, $v_r=0$, $v_{th}=1$, $I=1.5$")

    fhn_ax.plot(phases, fhn_kick, label=r"$\delta=10^{-3}$, crossing 4 cycles later")
    fhn_ax.plot(phases, fhn_half_kick, "--", label=r"$\delta=5\times10^{-4}$")
    fhn_ax.axhline(0, color="black", lw=0.8)
    fhn_ax.set_title(r"FHN: $I=0.5$, $a=0.7$, $b=0.8$, $\eta=0.08$")
    for ax in (lif_ax, fhn_ax):
        ax.set(xlabel=r"Phase $\phi$ (cycles)", ylabel="PRC $R$ (cycles / voltage unit)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "2c_prcs.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("hw3_results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    phases = np.linspace(0, 1, 201, endpoint=False)
    lif_kick, lif_formula = lif_prc(phases)
    period_fhn, fhn_kick = fhn_prc(phases, KICK_SIZE, RELAXATION_CYCLES)
    _, fhn_half_kick = fhn_prc(phases, KICK_SIZE / 2, RELAXATION_CYCLES)
    report_prcs(phases, period_fhn, lif_kick, lif_formula, fhn_kick, fhn_half_kick)
    np.savetxt(
        args.output_dir / "2c_prcs.csv",
        np.column_stack((phases, lif_kick, lif_formula, fhn_kick, fhn_half_kick)),
        delimiter=",", header="phase,lif_kick,lif_formula,fhn_kick,fhn_half_kick", comments="",
    )
    plot_prcs(phases, lif_kick, lif_formula, fhn_kick, fhn_half_kick, args.output_dir)


if __name__ == "__main__":
    main()
