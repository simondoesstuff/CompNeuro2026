import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar


def main():
    R = 1.0
    tau_m = 10.0
    tau_s = 2.0
    u_rest = 0.0
    u_th = 1.0

    t_peak_exact = (tau_m * tau_s / (tau_m - tau_s)) * np.log(tau_m / tau_s)
    I_crit_exact = (u_th / R) * (tau_m / tau_s) ** (tau_m / (tau_m - tau_s))

    def lif_ode(t, u, I0, tau_m=tau_m, tau_s=tau_s, R=R):
        return (-u + R * I0 * np.exp(-t / tau_s)) / tau_m

    def get_numerical_peak(I0, tau_m=tau_m, tau_s=tau_s):
        sol = solve_ivp(
            lif_ode,
            (0.0, 40.0),
            [u_rest],
            args=(I0, tau_m, tau_s, R),
            dense_output=True,
            rtol=1e-10,
            atol=1e-10,
        )
        res = minimize_scalar(
            lambda t: -sol.sol(t)[0], bounds=(0.0, 20.0), method="bounded"
        )
        return res.x, -res.fun

    I_crit_num = brentq(
        lambda I0: get_numerical_peak(I0)[1] - u_th, 1.0, 20.0, xtol=1e-9
    )
    t_peak_num, u_peak_at_crit = get_numerical_peak(I_crit_num)

    amplitudes = {
        "Below (0.8x)": 0.8 * I_crit_exact,
        "At (1.0x)": 1.0 * I_crit_exact,
        "Above (1.2x)": 1.2 * I_crit_exact,
    }

    t_span = (0.0, 40.0)
    t_eval = np.linspace(t_span[0], t_span[1], 4000)
    sim_results = {}

    for label, I0 in amplitudes.items():
        sol = solve_ivp(
            lif_ode,
            t_span,
            [u_rest],
            args=(I0, tau_m, tau_s, R),
            t_eval=t_eval,
            rtol=1e-9,
            atol=1e-9,
        )
        t_pk, u_pk_num = get_numerical_peak(I0)
        sim_results[label] = {
            "I0": I0,
            "t": sol.t,
            "u": sol.y[0],
            "u_peak_num": u_pk_num,
        }

    print("-" * 66)
    print(f"{'t_peak (ms)':<25} | {t_peak_num:<12.4f}")
    print(f"{'I_crit,0 (mA)':<25} | {I_crit_num:<12.4f}")

    print("\n" + "=" * 66)

    print(f"{'Condition':<15} | {'I_0 (mA)':<10} | {'Num. Peak (mV)':<15}")
    print("-" * 66)
    for label, data in sim_results.items():
        print(f"{label:<15} | {data['I0']:<10.4f} | {data['u_peak_num']:<15.4f}")

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = {
        "Below (0.8x)": "#1f77b4",
        "At (1.0x)": "#ff7f0e",
        "Above (1.2x)": "#2ca02c",
    }
    for label, data in sim_results.items():
        ax1.plot(
            data["t"],
            data["u"],
            label=f"{label}: $I_0 = {data['I0']:.4f}$ mA",
            color=colors[label],
            lw=2,
        )

    ax1.axhline(
        u_th,
        color="crimson",
        linestyle="--",
        lw=1.5,
        label=r"Threshold $u_{\mathrm{th}} = 1.0000$ mV",
    )
    ax1.axvline(
        t_peak_exact,
        color="gray",
        linestyle=":",
        lw=1.2,
        label=f"$t_{{\\mathrm{{peak}}}} = {t_peak_exact:.4f}$ ms",
    )
    ax1.set_xlabel("Time $t$ (ms)")
    ax1.set_ylabel("Membrane Potential $u(t)$ (mV)")
    ax1.set_title("LIF Voltage Traces for Exponential Pulse Inputs")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 40)
    ax1.set_ylim(0, 1.3)

    tau_s_range = np.logspace(np.log10(0.5), np.log10(100.0), 500)
    I_crit_curve = (u_th / R) * (tau_m / tau_s_range) ** (tau_m / (tau_m - tau_s_range))

    ax2.loglog(
        tau_s_range,
        I_crit_curve,
        color="#008080",
        lw=2.5,
        label=r"Exact $I_{\mathrm{crit},0}(\tau_s)$",
    )
    ax2.scatter(
        [tau_s],
        [I_crit_exact],
        color="red",
        zorder=5,
        s=60,
        label=f"Test Point ($\\tau_s = {tau_s:.1f}$ ms)",
    )

    ax2.set_xlabel(r"Synaptic Decay Time Constant $\tau_s$ (ms)")
    ax2.set_ylabel(r"Critical Injected Current $I_{\mathrm{crit},0}$ (mA)")
    ax2.set_title(r"Strength-Duration Curve on Log-Log Axes")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
