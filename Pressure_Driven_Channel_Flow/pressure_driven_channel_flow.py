import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pressure-driven flow between stationary parallel plates
# Momentum equation: mu*u'' = dp/dx
# No-slip boundaries: u(-h)=u(h)=0
mu = 1.0
pressure_gradient = -2.0
h = 1.0

# Characteristic scales
velocity_scale = max(
    abs(-pressure_gradient * h**2 / (2 * mu)), 1e-12
)
residual_scale = max(abs(pressure_gradient / mu), 1e-12)


def exact_velocity(y):
    return pressure_gradient * (y**2 - h**2) / (2 * mu)


def derivative(output, coordinate):
    return torch.autograd.grad(
        output,
        coordinate,
        torch.ones_like(output),
        create_graph=True,
    )[0]


class StokesNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 30), nn.Tanh(),
            nn.Linear(30, 30), nn.Tanh(),
            nn.Linear(30, 30), nn.Tanh(),
            nn.Linear(30, 1),
        )

    def forward(self, y):
        s = y / h
        raw_velocity = self.net(s)
        # Hard constraint: velocity is exactly zero at both walls.
        return velocity_scale * (1 - s**2) * raw_velocity


model = StokesNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=5000, gamma=0.5
)

y = torch.linspace(-h, h, 80, device=device).reshape(-1, 1)
y.requires_grad_(True)

# Training
losses = []
best_losses = []
best_loss = float("inf")
best_state = None
print(f"Training pressure-driven flow PINN on {device}...\n")

for epoch in range(10000):
    optimizer.zero_grad()
    y.grad = None

    velocity = model(y)
    velocity_y = derivative(velocity, y)
    velocity_yy = derivative(velocity_y, y)

    residual = velocity_yy - pressure_gradient / mu
    loss = torch.mean((residual / residual_scale) ** 2)

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

    if (epoch + 1) % 2000 == 0:
        print(f"Epoch {epoch + 1}: Loss = {current_loss:.2e}")

model.load_state_dict(best_state)
print(f"Best training loss: {best_loss:.2e}")

# Evaluation
y_test = torch.linspace(-h, h, 200, device=device).reshape(-1, 1)
with torch.no_grad():
    velocity_pred = model(y_test).cpu().numpy().flatten()

y_np = y_test.cpu().numpy().flatten()
velocity_exact = exact_velocity(y_np)
max_error = np.max(np.abs(velocity_pred - velocity_exact))
relative_l2 = np.linalg.norm(velocity_pred - velocity_exact) / np.linalg.norm(
    velocity_exact
)
wall_error = max(abs(velocity_pred[0]), abs(velocity_pred[-1]))

print("\n=== Results ===")
print(f"Maximum error:       {max_error:.3e}")
print(f"Relative L2 error:   {relative_l2:.3e}")
print(f"Maximum wall error:  {wall_error:.3e}")
print(f"Centerline velocity: {velocity_pred[len(velocity_pred) // 2]:.6f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(
    velocity_exact, y_np, "b-", linewidth=2.5, label="Exact"
)
axes[0].plot(
    velocity_pred, y_np, "r--", linewidth=2, label="PINN"
)
axes[0].set_xlabel("Velocity u(y)")
axes[0].set_ylabel("Channel position y")
axes[0].set_title("Pressure-Driven Channel Flow")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].semilogy(losses, color="gray", alpha=0.4, label="Training loss")
axes[1].semilogy(best_losses, color="darkgreen", label="Best loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Normalized PDE loss")
axes[1].set_title("Training Convergence")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("chapter4_stokes_flow.png", dpi=200, bbox_inches="tight")
plt.show()
