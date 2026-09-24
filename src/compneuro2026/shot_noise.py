import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, skew

V_mean = 10.0  # mV
tau = 0.020  # s (20 ms)
T_total = 200.0  # total simulation duration (s)
dt = 0.0001  # step size: 0.1 ms
n_steps = int(T_total / dt)
decay = np.exp(-dt / tau)

nu_tau_values = [2, 20, 200]
results = {}

for nutau in nu_tau_values:
    w = V_mean / nutau
    nu = nutau / tau

    events = np.random.poisson(nu * dt, size=n_steps)

    # Exact exponential integration
    V = np.zeros(n_steps)
    for t in range(1, n_steps):
        V[t] = V[t - 1] * decay + w * events[t]

    # Discard transient initial 5*tau
    warmup = int(5 * tau / dt)
    V_steady = V[warmup:]

    results[nutau] = {
        "V": V_steady,
        "mean": np.mean(V_steady),
        "var": np.var(V_steady),
        "skew": skew(V_steady),
    }

print(f"{'nu*tau':<8} | {'Stat':<10} | {'Theoretical':<12} | {'Measured':<12}")
print("-" * 50)
for nutau in nu_tau_values:
    th_var = (nutau * (V_mean / nutau) ** 2) / 2
    th_skew = (2 * np.sqrt(2)) / (3 * np.sqrt(nutau))
    print(
        f"{nutau:<8} | {'Mean (mV)':<10} | {10.0:<12.3f} | {results[nutau]['mean']:<12.3f}"
    )
    print(
        f"{'':<8} | {'Var (mV²)':<10} | {th_var:<12.4f} | {results[nutau]['var']:<12.4f}"
    )
    print(
        f"{'':<8} | {'Skewness':<10} | {th_skew:<12.4f} | {results[nutau]['skew']:<12.4f}"
    )
    print("-" * 50)

# Histogram overlay for extreme cases: nu*tau = 2 and 200
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, nutau in zip(axes, [2, 200]):
    V_data = results[nutau]["V"]
    mu = np.mean(V_data)
    sigma = np.std(V_data)

    count, bins, _ = ax.hist(
        V_data, bins=80, density=True, alpha=0.6, color="teal", label="Simulation"
    )
    x = np.linspace(bins[0], bins[-1], 300)
    ax.plot(x, norm.pdf(x, mu, sigma), "r--", lw=2, label="Matched Gaussian")
    ax.set_title(f"$\\nu\\tau = {nutau}$ (w = {V_mean / nutau} mV)")
    ax.set_xlabel("Membrane Potential $V$ (mV)")
    ax.set_ylabel("Probability Density")
    ax.legend()

plt.tight_layout()
plt.show()
