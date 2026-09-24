import matplotlib.pyplot as plt
import numpy as np

mu = 1.0
theta = 1.0
sigma2_values = [0.25, 1.0, 4.0]

dt = 0.001  # Time step
t_transient = 100.0  # Burn-in time to discard
t_eval = 2000.0  # Evaluation time after transient
t_total = t_transient + t_eval

n_steps = int(t_total / dt)
n_burn = int(t_transient / dt)


# Analytical density function p(u)
def theoretical_p(u, s2):
    nu = mu / theta
    p = np.zeros_like(u)
    # 0 <= u <= theta
    idx_pos = (u >= 0) & (u <= theta)
    p[idx_pos] = (nu / mu) * (1.0 - np.exp(mu * (u[idx_pos] - theta) / s2))
    # u < 0
    idx_neg = u < 0
    p[idx_neg] = (
        (nu / mu) * (1.0 - np.exp(-mu * theta / s2)) * np.exp(mu * u[idx_neg] / s2)
    )
    return p


nu_theory = mu / theta  # 1.0


def frac_theory(s2):
    return (s2 / (mu * theta)) * (1.0 - np.exp(-mu * theta / s2))


fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

print(
    f"{'sigma^2':<8} | {'nu (Meas)':<10} | {'nu (Theory)':<11} | {'Frac < 0 (Meas)':<16} | {'Frac < 0 (Theory)':<17}"
)
print("-" * 72)

np.random.seed(42)

for ax, s2 in zip(axes, sigma2_values):
    noise_std = np.sqrt(2.0 * s2 * dt)
    dW = np.random.normal(0.0, noise_std, size=n_steps)

    u_traj = np.zeros(n_steps)
    u_curr = 0.0
    spikes_after_burn = 0

    for step in range(n_steps):
        u_curr += mu * dt + dW[step]
        if u_curr >= theta:
            if step >= n_burn:
                spikes_after_burn += 1
            u_curr = 0.0
        u_traj[step] = u_curr

    # Keep only post-transient trajectory
    u_post = u_traj[n_burn:]

    nu_meas = spikes_after_burn / t_eval
    frac_meas = np.mean(u_post < 0.0)
    f_pred = frac_theory(s2)

    print(
        f"{s2:<8.2f} | {nu_meas:<10.3f} | {nu_theory:<11.3f} | {frac_meas:<16.3f} | {f_pred:<17.3f}"
    )

    u_min = np.percentile(u_post, 0.1)
    bins = np.linspace(min(u_min, -3.0 * np.sqrt(s2)), theta, 80)
    ax.hist(
        u_post,
        bins=bins,
        density=True,
        alpha=0.6,
        color="steelblue",
        label="Simulation",
    )

    u_grid = np.linspace(bins[0], theta, 400)
    p_grid = theoretical_p(u_grid, s2)
    ax.plot(u_grid, p_grid, "r-", lw=2, label="Theory $p(u)$")

    ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_title(rf"$\sigma^2 = {s2}$")
    ax.set_xlabel("Voltage $u$")
    ax.legend()

axes[0].set_ylabel("Probability Density")
plt.tight_layout()
plt.show()
