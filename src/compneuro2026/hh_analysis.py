import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from compneuro2026.hh_sim import hh_sim

TMAX = 300.0
SKIPFRAC = 0.4
MIN_SPIKES = 2  # spikes in the tail window required


def fires(Id, gK=36.0):
    _, _, spk, _ = hh_sim(Id=Id, tmax=TMAX, gK=gK, skipfrac=SKIPFRAC)
    return len(spk) >= MIN_SPIKES


def bsearch_rheo(gK=36.0, lo=0.5, hi=10.0, tol=1e-4):
    while (hi - lo) > tol:
        mid = 0.5 * (lo + hi)
        if fires(mid, gK):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def fi_curve(rh, gK=36.0, n_points=35):
    I_vals = np.linspace(rh + 0.05, 2.0 * rh, n_points)
    rates = []
    for Id in I_vals:
        _, _, _, r = hh_sim(Id=Id, tmax=TMAX, gK=gK, skipfrac=SKIPFRAC)
        rates.append(r)
    return I_vals, np.array(rates)


def part_a(rh, gK=36.0):
    print("\nPart A")
    print(f"  gK = {gK}  rheobase = {rh:.3f}")
    print(
        f"  Criterion: >= {MIN_SPIKES} spikes in last "
        f"{int((1 - SKIPFRAC) * 100)} % of {TMAX:.0f} ms run"
    )

    Id_lo = rh - 0.02
    Id_hi = rh + 0.02

    t_lo, V_lo, spk_lo, r_lo = hh_sim(Id=Id_lo, tmax=TMAX, gK=gK, skipfrac=SKIPFRAC)
    t_hi, V_hi, spk_hi, r_hi = hh_sim(Id=Id_hi, tmax=TMAX, gK=gK, skipfrac=SKIPFRAC)

    print(
        f"  Id = {Id_lo:.3f}: {len(spk_lo)} spikes, rate = {r_lo:.1f} Hz  (sub-threshold)"
    )
    print(
        f"  Id = {Id_hi:.3f}: {len(spk_hi)} spikes, rate = {r_hi:.1f} Hz  (supra-threshold)"
    )

    Vlim = (-20, 130)
    tlim = (0, TMAX)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax1.plot(t_lo, V_lo, "C0", lw=0.8)
    ax1.set_title(f"Id = {Id_lo:.3f} µA/cm²  (below rheobase = {rh:.3f})", fontsize=9)
    ax2.plot(t_hi, V_hi, "C1", lw=0.8)
    ax2.set_title(f"Id = {Id_hi:.3f} µA/cm²  (above rheobase = {rh:.3f})", fontsize=9)

    for ax in (ax1, ax2):
        ax.set_xlim(tlim)
        ax.set_ylim(Vlim)
        ax.set_ylabel("V  (mV)")
    ax2.set_xlabel("t  (ms)")

    fig.suptitle(
        "HH Voltage Traces Near Rheobase  (gK = 36 mS/cm²)",
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig("hh_part_a.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved")
    return rh


def part_b(rh, I_vals, rates, gK=36.0):
    print("\nPart b")
    nonzero = rates[rates > 0]
    onset_rate = nonzero[0] if len(nonzero) else 0.0
    print(f"  Rheobase = {rh:.3f}")
    print(f"  Onset firing rate = {onset_rate:.3f} Hz")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(I_vals, rates, "C0o-", ms=4, lw=1.2, label=f"gK = {gK}")
    ax.axvline(rh, color="C0", ls="--", lw=0.8, label=f"rheobase = {rh:.3f}")
    ax.set_xlabel("Applied current  Id  (µA/cm²)")
    ax.set_ylabel("Firing rate  (Hz)")
    ax.set_title(f"f–I curve  (HH, gK = {gK} mS/cm²)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("hh_part_b.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved")
    return onset_rate


def part_c(I_vals_36, rates_36, rh_36, rh_30, I_vals_30, rates_30):
    print("\nPart c")
    onset_36 = rates_36[rates_36 > 0][0] if any(rates_36 > 0) else 0.0
    onset_30 = rates_30[rates_30 > 0][0] if any(rates_30 > 0) else 0.0

    print("\n  Rheobase comparison:")
    print(f"  {'gK':<14} {'Rheobase':<20} {'Onset rate (Hz)'}")
    print(f"  {'-' * 14} {'-' * 20} {'-' * 16}")
    print(f"  {'36':<14} {rh_36:<20.3f} {onset_36:.3f}")
    print(f"  {'30':<14} {rh_30:<20.3f} {onset_30:.3f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(I_vals_36, rates_36, "C0o-", ms=4, lw=1.2, label="gK = 36")
    ax.plot(I_vals_30, rates_30, "C1s-", ms=4, lw=1.2, label="gK = 30")
    ax.axvline(rh_36, color="C0", ls="--", lw=0.9, label=f"rheobase (36) = {rh_36:.3f}")
    ax.axvline(rh_30, color="C1", ls="--", lw=0.9, label=f"rheobase (30) = {rh_30:.3f}")
    ax.set_xlabel("Applied current  Id  (µA/cm²)")
    ax.set_ylabel("Firing rate  (Hz)")
    ax.set_title("f–I curves: gK = 36 vs 30 mS/cm²  (HH model)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("hh_part_c.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved")
    return rh_30, onset_30


def main():
    print("Computing rheobase for gK=36 ...")
    rh_36 = bsearch_rheo(gK=36.0)
    print(f"  rh_36 = {rh_36:.4f}")

    print("Computing rheobase for gK=30 ...")
    rh_30 = bsearch_rheo(gK=30.0)
    print(f"  rh_30 = {rh_30:.4f}")

    print("Computing f-I curve for gK=36 ...")
    I_36, rates_36 = fi_curve(rh_36, gK=36.0)

    print("Computing f-I curve for gK=30 ...")
    I_30, rates_30 = fi_curve(rh_30, gK=30.0)

    part_a(rh=rh_36, gK=36.0)
    part_b(rh_36, I_36, rates_36, gK=36.0)
    part_c(I_36, rates_36, rh_36, rh_30, I_30, rates_30)


if __name__ == "__main__":
    main()
