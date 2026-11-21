"""Independent sampling, quadrature, and ODE checks for HW6 results."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad

try:
    from .hw6_2c import EQUILIBRIUM, invariant, mean_flow
except ImportError:
    from hw6_2c import EQUILIBRIUM, invariant, mean_flow


def run(out):
    foraging = json.loads((out / "1c_results.json").read_text())
    learning = json.loads((out / "2c_results.json").read_text())
    errors = [abs(r["error_in_standard_errors"]) for r in foraging["simulations"]]
    assert max(errors) < 5, "Foraging estimate exceeds five standard errors"
    optimum = foraging["optimal_wait"]
    # The optimality equation reduces to exp(rho*T) = 1+rho*T+rho*tau/(1-p).
    optimality_error = abs(np.exp(0.5 * optimum) - (1 + 0.5 * optimum + 5))
    assert optimality_error < 1e-10

    quadrature_errors = []
    for point in [[0.2, 0.3], [0.8, 0.1], [0.47, 0.53]]:
        numerical = sum(quad(lambda s: (s - a) / (s * (1 - s))**2, a, b)[0]
                        for a, b in zip(EQUILIBRIUM, point))
        quadrature_errors.append(abs(numerical - invariant(point)))
    assert max(quadrature_errors) < 1e-10
    _, base = mean_flow()
    _, refined = mean_flow(rtol=1e-12)
    flow_error = float(np.max(abs(base - refined)))
    assert flow_error < 1e-7
    assert learning["mean_flow"]["K_variation"] < 1e-7

    # Enumerate action/outcome probabilities independently of simulation code.
    # Baselines must cancel from the expected score-function updates.
    gradient_error = 0.0
    for x, y in [[0.2, 0.3], [0.7, 0.8], EQUILIBRIUM]:
        for baseline in [-1, 0, 1.25]:
            gradient = np.zeros(2)
            for a in [0, 1]:
                for d in [0, 1]:
                    make = {(0, 0): 0.6, (0, 1): 0.4,
                            (1, 0): 0.25, (1, 1): 0.4}[a, d]
                    prob_actions = (x if a else 1 - x) * (y if d else 1 - y)
                    for made in [0, 1]:
                        prob = prob_actions * (make if made else 1 - make)
                        advantage = (2 + a) * made - baseline
                        gradient += prob * advantage * np.array([a - x, -(d - y)])
            expected = np.array([0.85 * x * (1 - x) * (y - 9 / 17),
                                 -0.85 * y * (1 - y) * (x - 8 / 17)])
            gradient_error = max(gradient_error, float(np.max(abs(gradient - expected))))
    assert gradient_error < 1e-14
    result = {
        "foraging_absolute_z_scores": errors,
        "optimality_equation_residual": optimality_error,
        "K_quadrature_max_error": float(max(quadrature_errors)),
        "mean_flow_tighter_tolerance_max_change": flow_error,
        "mean_flow_K_variation": learning["mean_flow"]["K_variation"],
        "enumerated_expected_gradient_max_error": gradient_error,
    }
    (out / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("hw6_results"))
    run(parser.parse_args().output_dir)
