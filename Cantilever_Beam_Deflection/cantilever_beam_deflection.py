import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Euler-Bernoulli cantilever beam: EI*w''''=q
# BCs: w(0)=w'(0)=0 and w''(L)=w'''(L)=0
E = 200e9
I = 8.33e-6
q = 1000.0
L = 1.0
EI = E * I
w_scale = q * L**4 / EI


def exact_deflection(x):
    return q * (x**4 - 4 * L * x**3 + 6 * L**2 * x**2) / (24 * EI)


def derivative(output, coordinate):
    return torch.autograd.grad(
        output,
        coordinate,
        torch.ones_like(output),
        create_graph=True,
    )[0]


class BeamNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 30), nn.Tanh(),
            nn.Linear(30, 30), nn.Tanh(),
            nn.Linear(30, 30), nn.Tanh(),
            nn.Linear(30, 1),
        )

    def forward(self, s):
        # Dimensionless deflection v=w/w_scale; s² enforces v(0)=v'(0)=0.
        return s**2 * self.net(s)


model = BeamNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=4000, gamma=0.5
)

s = torch.linspace(0, 1, 60, device=device).reshape(-1, 1)
s.requires_grad_(True)
sL = torch.ones(1, 1, device=device, requires_grad=True)

# Dimensionless equation: v''''=1
losses, best_losses = [], []
best_loss = float("inf")
best_state = None
print(f"Training nondimensional beam PINN on {device}...\n")

for epoch in range(8000):
    optimizer.zero_grad()
    s.grad = None
    sL.grad = None

    v = model(s)
    v1 = derivative(v, s)
    v2 = derivative(v1, s)
    v3 = derivative(v2, s)
    v4 = derivative(v3, s)
    loss_pde = torch.mean((v4 - 1) ** 2)

    # Free-end moment and shear: v''(1)=v'''(1)=0
    vL = model(sL)
    v1L = derivative(vL, sL)
    v2L = derivative(v1L, sL)
    v3L = derivative(v2L, sL)
    loss_free = torch.mean(v2L**2 + v3L**2)

    loss = loss_pde + 10 * loss_free
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
s_test = torch.linspace(0, 1, 200, device=device).reshape(-1, 1)
with torch.no_grad():
    v_pred = model(s_test).cpu().numpy().flatten()

x = s_test.cpu().numpy().flatten() * L
w_pred = w_scale * v_pred
w_exact = exact_deflection(x)
max_error = np.max(np.abs(w_pred - w_exact))
relative_error = max_error / np.max(np.abs(w_exact))

print("\n=== Results ===")
print(f"Tip deflection (exact): {w_exact[-1] * 1e3:.6f} mm")
print(f"Tip deflection (PINN):  {w_pred[-1] * 1e3:.6f} mm")
print(f"Maximum error:          {max_error * 1e6:.3f} μm")
print(f"Relative max error:     {relative_error * 100:.4f}%")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(x * 1e3, w_exact * 1e3, "b-", linewidth=2.5, label="Exact")
axes[0].plot(x * 1e3, w_pred * 1e3, "r--", linewidth=2, label="PINN")
axes[0].set_xlabel("Position x [mm]")
axes[0].set_ylabel("Deflection w [mm]")
axes[0].set_title("Cantilever Beam Deflection")
axes[0].invert_yaxis()
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].semilogy(losses, color="gray", alpha=0.4, label="Training loss")
axes[1].semilogy(best_losses, color="darkgreen", label="Best loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Dimensionless loss")
axes[1].set_title("Training Convergence")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("chapter6_beam_deflection.png", dpi=200, bbox_inches="tight")
plt.show()
