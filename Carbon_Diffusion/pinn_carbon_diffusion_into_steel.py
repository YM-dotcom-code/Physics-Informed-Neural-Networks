import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc


torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fick's second law: C_t=D*C_xx
D = 1e-11              # Diffusivity [m²/s]
L = 0.002              # Modeled depth [m]
t_final = 3600.0       # Final time [s]
t_star_final = D * t_final / L**2


def analytical_concentration(x, t):
    if t == 0:
        return np.zeros_like(x)
    return erfc(x / (2 * np.sqrt(D * t)))


def derivative(output, coordinate):
    return torch.autograd.grad(
        output,
        coordinate,
        torch.ones_like(output),
        create_graph=True,
    )[0]


class DiffusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 40), nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
            nn.Linear(40, 1),
        )

    def forward(self, x, tau):
        # sqrt(tau) helps represent the diffusion boundary-layer scaling.
        features = torch.cat(
            [x, tau, torch.sqrt(tau + 1e-6)], dim=1
        )
        # Concentration remains in [0,1], and C(1,t)=0 exactly.
        return (1 - x) * torch.sigmoid(self.net(features))


model = DiffusionNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=5000, gamma=0.5
)

# Quasi-random points plus boundary-layer-biased points.
sobol = torch.quasirandom.SobolEngine(2, scramble=True, seed=42)
points_uniform = sobol.draw(500).to(device)
points_biased = sobol.draw(500).to(device)
x_int = torch.cat(
    [points_uniform[:, :1], points_biased[:, :1] ** 2], dim=0
)
tau_int = torch.cat(
    [points_uniform[:, 1:], points_biased[:, 1:] ** 2], dim=0
)
tau_int = 1e-4 + (1 - 1e-4) * tau_int
x_int.requires_grad_(True)
tau_int.requires_grad_(True)

# Exclude the incompatible corner (x,tau)=(0,0) from both condition sets.
tau_bc = torch.linspace(1e-4, 1, 250, device=device).reshape(-1, 1)
x_bc = torch.zeros_like(tau_bc)
x_ic = torch.linspace(1e-3, 1, 250, device=device).reshape(-1, 1)
tau_ic = torch.zeros_like(x_ic)

losses, best_losses = [], []
best_loss = float("inf")
best_state = None
print(f"Training carbon-diffusion PINN on {device}...\n")

for epoch in range(12000):
    optimizer.zero_grad()
    x_int.grad = None
    tau_int.grad = None

    concentration = model(x_int, tau_int)
    C_tau = derivative(concentration, tau_int)
    C_x = derivative(concentration, x_int)
    C_xx = derivative(C_x, x_int)

    # tau=t/t_final, so C_tau=t*_final*C_xx.
    residual = C_tau - t_star_final * C_xx
    loss_pde = torch.mean(residual**2)
    loss_surface = torch.mean((model(x_bc, tau_bc) - 1) ** 2)
    loss_initial = torch.mean(model(x_ic, tau_ic) ** 2)
    loss = loss_pde + 20 * loss_surface + 20 * loss_initial

    loss.backward()
    optimizer.step()
    scheduler.step()

    current_loss = loss.item()
    losses.append(current_loss)
    if current_loss < best_loss:
        best_loss = current_loss
        best_state = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
    best_losses.append(best_loss)

    if (epoch + 1) % 3000 == 0:
        print(f"Epoch {epoch + 1}: Loss = {current_loss:.3e}")

model.load_state_dict(best_state)
print(f"Best training loss: {best_loss:.3e}")

# Evaluation
sample_times = [900, 1800, 3600]
x_plot = np.linspace(0, L, 300)
x_star = torch.tensor(
    x_plot / L, dtype=torch.float32, device=device
).reshape(-1, 1)

results = []
print("\n=== Results ===")
for physical_time in sample_times:
    tau_value = physical_time / t_final
    tau = torch.full_like(x_star, tau_value)
    with torch.no_grad():
        predicted = model(x_star, tau).cpu().numpy().flatten()
    exact = analytical_concentration(x_plot, physical_time)
    max_error = np.max(np.abs(predicted - exact))
    relative_l2 = np.linalg.norm(predicted - exact) / np.linalg.norm(exact)
    results.append((physical_time, predicted, exact, max_error, relative_l2))
    print(
        f"t={physical_time:4d} s: max error={max_error:.4f}, "
        f"relative L2={relative_l2:.4f}"
    )

# Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = ["blue", "red", "green"]
for (physical_time, predicted, exact, _, _), color in zip(results, colors):
    axes[0].plot(
        x_plot * 1e3, exact, color=color, linewidth=2.5,
        label=f"Exact {physical_time} s",
    )
    axes[0].plot(
        x_plot * 1e3, predicted, "--", color=color, linewidth=2,
        label=f"PINN {physical_time} s",
    )

axes[0].set_xlabel("Depth [mm]")
axes[0].set_ylabel("Normalized carbon concentration")
axes[0].set_title("Carbon Diffusion into Steel")
axes[0].legend(ncol=2, fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].semilogy(losses, color="gray", alpha=0.4, label="Training loss")
axes[1].semilogy(best_losses, color="darkgreen", label="Best loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Normalized loss")
axes[1].set_title("Training Convergence")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("chapter7_diffusion.png", dpi=200, bbox_inches="tight")
plt.show()
