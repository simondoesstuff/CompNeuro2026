"""Independent convolution and convergence checks for the HW5 experiments.

Run after the four problem scripts, from the repository root.
"""

import json
from pathlib import Path

import numpy as np
from hw5_1d import simulate as oscillation
from hw5_2d import simulate as rivalry
from hw5_3d import simulate as bump
from hw5_3d import widths
from hw5_4c import simulate as turing


def run():
    out = Path("hw5_results")
    result = {}
    base = json.loads((out / "1d_results.json").read_text())["epsilon_sweep"][0]
    refined, _, _ = oscillation(10, 0.005, rtol=1e-11)
    errors = {
        key: abs(refined[key] - base[key]) for key in ["period", "amplitude", "lag_deg"]
    }
    assert (
        errors["period"] < 1e-5
        and errors["amplitude"] < 1e-6
        and errors["lag_deg"] < 1e-3
    )
    result["oscillation_tighter_tolerance_absolute_changes"] = errors
    base_rows = json.loads((out / "2d_results.json").read_text())
    rivalry_errors = []
    for index in [2, 3]:
        base = base_rows[index]
        refined, _ = rivalry(base["il"], base["w"], rtol=1e-10, max_step=0.25)
        errors = {key: abs(refined[key] - base[key]) for key in ["mean_l", "mean_r"]}
        assert max(errors.values()) < 1e-4
        rivalry_errors.append(
            {"il": base["il"], "w": base["w"], "absolute_changes": errors}
        )
    result["rivalry_tighter_tolerance_and_step"] = rivalry_errors
    _, wide = widths(0.3)
    _, _, x, _, convolve = bump(0.3, wide, end=0.02)
    rng = np.random.default_rng(1)
    f = rng.normal(size=len(x))
    lag = np.arange(-(len(x) - 1), len(x)) * 0.02
    kernel = 2 * np.exp(-abs(lag)) - np.exp(-abs(lag) / 2)
    direct = 0.02 * np.convolve(f, kernel, "full")[len(x) - 1 : 2 * len(x) - 1]
    fft_error = float(max(abs(direct - convolve(f))))
    assert fft_error < 1e-12
    result["fft_vs_direct_convolution_max_error"] = fft_error
    t, a, *_ = bump(0.3, wide + 0.05, dt=0.01, dx=0.01)
    mask = (t >= 1) & (t <= 8)
    rate = float(np.polyfit(t[mask], np.log(abs(a[mask] - wide)), 1)[0])
    result["bump_refined_dx_dt_001"] = {
        "final_width": float(a[-1]),
        "raw_fit_rate": rate,
    }
    base = json.loads((out / "3d_results.json").read_text())["relaxation"][1]
    assert abs(a[-1] - base["sigmoid_width"]) < 1e-3
    assert abs(rate - base["raw_fit_rate"]) < 0.005
    noise = np.random.default_rng(4370).normal(0, 1e-3, 600)
    refined_u, refined = turing(2, noise, dt=0.01)
    base_u = np.loadtxt(out / "4c_final_gamma_2.csv", delimiter=",", skiprows=1)[:, 1]
    error = float(max(abs(refined_u - base_u)))
    assert error < 1e-5 and refined["mode"] == 7
    result["turing_half_dt_max_profile_change"] = error
    (out / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
