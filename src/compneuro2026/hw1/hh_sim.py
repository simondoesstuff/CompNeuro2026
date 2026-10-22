"""Classical Hodgkin-Huxley model under constant current injection.

Python port of hh_sim.m. Same parameters, same shifted potential
convention of the 1952 papers, so rest is near V = 0 and the spike peak
near V = 100.

Call it and capture what you need:

    t, V, spk, rate = hh_sim()

Running the file directly (python hh_sim.py) prints the spike count and
rate at the default Id = 7. There is no MATLAB-style ans here, but t is
30001 elements of 0 to 300 in steps of 0.01, so avoid echoing the whole
return value in a notebook cell.

Regression checks with default parameters:
    Id = 7 gives sustained firing at 58.3 Hz
    Id = 5 gives no sustained firing
The 58.3 is solver independent: BDF, LSODA, Radau and RK45 all agree to
three digits, so a port that lands more than a Hz away has a real bug,
most likely a sign in an exponent. The removable singularities in
alpha_m and alpha_n at V = 25 and V = 10 are patched below; if you
rewrite them, check that alpha_m(25) and alpha_n(10) come out finite.

The reported rheobase depends on how you define repetitive firing: both
skipfrac and the spike count you demand of it will move the answer,
because the underlying bifurcation is subcritical. State the criterion
you used rather than treating the number as absolute.
"""

import numpy as np
from scipy.integrate import solve_ivp

__all__ = ["hh_sim"]


# ---- gating rate functions (ms^-1), with removable singularities patched ----
def _am(V):
    return 1.0 if abs(V - 25) < 1e-6 else 0.1 * (25 - V) / (np.exp((25 - V) / 10) - 1)


def _bm(V):
    return 4 * np.exp(-V / 18)


def _ah(V):
    return 0.07 * np.exp(-V / 20)


def _bh(V):
    return 1 / (np.exp((30 - V) / 10) + 1)


def _an(V):
    return 0.1 if abs(V - 10) < 1e-6 else 0.01 * (10 - V) / (np.exp((10 - V) / 10) - 1)


def _bn(V):
    return 0.125 * np.exp(-V / 80)


def hh_sim(Id=7.0, tmax=300.0, gK=36.0, gNa=120.0, skipfrac=0.4):
    """Integrate the HH model under constant current.

    Parameters
    ----------
    Id : injected current density (uA/cm^2). Defaults to 7, above
         rheobase, so hh_sim() with no arguments gives repetitive
         firing.
    tmax : simulation duration (ms).
    gK, gNa : maximal conductances (mS/cm^2).
    skipfrac : fraction of the trace discarded before counting spikes, so
        that the onset transient is not mistaken for repetitive firing.

    Returns
    -------
    t, V : time vector (ms) and voltage trace (mV).
    spk : spike times, upward crossings of 40 mV.
    rate : mean firing rate in Hz over the counted window, 0 if fewer
        than two spikes.
    """
    C, gL, ENa, EK, EL = 1.0, 0.3, 115.0, -12.0, 10.6

    def rhs(_, y):
        V, m, h, n = y
        INa = gNa * m**3 * h * (V - ENa)
        IK = gK * n**4 * (V - EK)
        IL = gL * (V - EL)
        return [
            (Id - INa - IK - IL) / C,
            _am(V) * (1 - m) - _bm(V) * m,
            _ah(V) * (1 - h) - _bh(V) * h,
            _an(V) * (1 - n) - _bn(V) * n,
        ]

    V0 = 0.0  # steady state gating at rest
    y0 = [
        V0,
        _am(V0) / (_am(V0) + _bm(V0)),
        _ah(V0) / (_ah(V0) + _bh(V0)),
        _an(V0) / (_an(V0) + _bn(V0)),
    ]

    t = np.arange(0, tmax + 1e-9, 0.01)
    sol = solve_ivp(rhs, [0, tmax], y0, method="BDF", t_eval=t,
                    rtol=1e-8, atol=1e-10, max_step=0.05)
    V = sol.y[0]

    i0 = max(1, int(round(skipfrac * len(t))))
    up = np.flatnonzero((V[i0:-1] < 40) & (V[i0 + 1:] >= 40)) + i0
    spk = t[up]

    rate = 1000 * (len(spk) - 1) / (spk[-1] - spk[0]) if len(spk) >= 2 else 0.0
    return t, V, spk, rate


if __name__ == "__main__":
    t, V, spk, rate = hh_sim()
    print(f"Id = 7: {len(spk)} spikes, rate = {rate:.1f} Hz")
