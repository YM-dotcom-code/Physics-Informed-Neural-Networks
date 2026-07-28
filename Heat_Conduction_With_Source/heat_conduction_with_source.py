import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Steady 1D conduction with uniform internal heat generation
# k*T''+Q=0, T(0)=T0, T(L)=TL
k = 50.0
Q = 1000.0
L = 1.0
T0, TL = 100.0, 200.0

# T=T0+T_scale*theta makes all trained quantities O(1).
T_scale = max(abs(TL - T0), 1.0)
theta_left = 0.0
theta_right = (TL - T0) / T_scale
source = Q * L**2 / (k * T_scale)


def exact_temperature(x):
    return -(Q / (2 * k)) * x**2 + (
        (TL - T0) / L + Q * L / (2 * k)
    ) * x + T0


def derivative(output, coordinate):
    return torch.autograd.grad(
        output,
        coordinate,
        torch.ones_like(output),
        create_graph=True,
    )[0]


class HeatNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 30), nn.Tanh(),
            nn.Linear(30, 30), nn.Tanh(),
            nn.Linear(30, 30), nn.Tanh(),
            nn.Linear(30, 1),
        )

    def forward(self, s):
        return self.net(s)


def train_heat_pinn(boundary_weight, epochs=6000):
    # Identical initial weights make the comparison fair.
    torch.manual_seed(42)
    model = HeatNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3000, gamma=0.5
    )

    s = torch.linspace(0, 1, 80, device=device).reshape(-1, 1)
    s.requires_grad_(True)
    s_bc = torch.tensor([[0.0], [1.0]], device=device)
    theta_bc = torch.tensor(
        [[theta_left], [theta_right]], device=device
    )

    losses, best_losses = [], []
    best_loss = float("inf")
    best_state = None

    for _ in range(epochs):
        optimizer.zero_grad()
        s.grad = None

        theta = model(s)
        theta_s = derivative(theta, s)
        theta_ss = derivative(theta_s, s)
        loss_pde = torch.mean((theta_ss + source) ** 2)

        theta_pred_bc = model(s_bc)
        loss_bc = torch.mean((theta_pred_bc - theta_bc) ** 2)
        loss = loss_pde + boundary_weight * loss_bc

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

    model.load_state_dict(best_state)

    s_test = torch.linspace(0, 1, 200, device=device).reshape(-1, 1)
    s_test.requires_grad_(True)
    theta = model(s_test)
    theta_s = derivative(theta, s_test)
    theta_ss = derivative(theta_s, s_test)

    temperature = (T0 + T_scale * theta).detach().cpu().numpy().flatten()
    x = s_test.detach().cpu().numpy().flatten() * L
    exact = exact_temperature(x)
    physical_residual = (
        k * T_scale / L**2 * theta_ss + Q
    ).detach().cpu().numpy().flatten()

    with torch.no_grad():
        predicted_bc = T0 + T_scale * model(s_bc)
    exact_bc = torch.tensor([[T0], [TL]], device=device)

    return {
        "model": model,
        "x": x,
        "temperature": temperature,
        "exact": exact,
        "residual": physical_residual,
        "losses": losses,
        "best_losses": best_losses,
        "max_error": np.max(np.abs(temperature - exact)),
        "boundary_error": torch.max(
            torch.abs(predicted_bc - exact_bc)
        ).item(),
        "mean_residual": np.mean(np.abs(physical_residual)),
        "best_loss": best_loss,
    }


print(f"Comparing normalized BC weights on {device}...\n")
weak = train_heat_pinn(boundary_weight=1)
strong = train_heat_pinn(boundary_weight=100)

for weight, result in [(1, weak), (100, strong)]:
    print(
        f"lambda_bc={weight:>3}: "
        f"max error={result['max_error']:.4f} °C, "
        f"boundary error={result['boundary_error']:.4f} °C, "
        f"mean |residual|={result['mean_residual']:.3e}"
    )

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].plot(weak["x"], weak["exact"], "b-", linewidth=2.5, label="Exact")
axes[0].plot(
    strong["x"], strong["temperature"], "r--", linewidth=2,
    label="PINN (lambda=100)",
)
axes[0].plot(
    weak["x"], weak["temperature"], "g:", linewidth=2,
    label="PINN (lambda=1)",
)
axes[0].set(xlabel="x [m]", ylabel="T [°C]", title="Temperature Profile")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].semilogy(
    weak["x"], np.maximum(np.abs(weak["residual"]), 1e-12),
    "g-", label="lambda=1",
)
axes[1].semilogy(
    strong["x"], np.maximum(np.abs(strong["residual"]), 1e-12),
    "r-", label="lambda=100",
)
axes[1].set(
    xlabel="x [m]", ylabel="|k*T''+Q|",
    title="Physical PDE Residual",
)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].semilogy(weak["losses"], "g-", alpha=0.35)
axes[2].semilogy(weak["best_losses"], "g-", label="lambda=1")
axes[2].semilogy(strong["losses"], "r-", alpha=0.35)
axes[2].semilogy(strong["best_losses"], "r-", label="lambda=100")
axes[2].set(xlabel="Epoch", ylabel="Normalized loss", title="Convergence")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("chapter5_heat_conduction.png", dpi=200, bbox_inches="tight")
plt.show()
