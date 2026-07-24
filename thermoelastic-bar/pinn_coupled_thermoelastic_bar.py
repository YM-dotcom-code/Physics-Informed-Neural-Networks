import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Coupled 1D thermoelastic problem
# Heat: k*T''=0
# Stress: sigma=E*(u' - alpha*(T-T_ref))
# Equilibrium: sigma'=0, hence u'' - alpha*T'=0
k = 50.0
E = 200e9
alpha = 12e-6
L = 1.0
T0, TL = 100.0, 500.0
T_ref = 100.0

# Signed physics and positive normalization scales
dT = TL - T0
T_scale = max(abs(dT), 1.0)
strain_scale = max(
    abs(alpha * (T0 - T_ref)),
    abs(alpha * (TL - T_ref)),
    1e-12,
)
u_scale = strain_scale * L
curvature_scale = strain_scale / L


def exact_T(x):
    return T0 + dT * x / L


def exact_u(x):
    return alpha * (T0 - T_ref) * x + alpha * dT * x**2 / (2 * L)


def derivative(y, x):
    return torch.autograd.grad(
        y, x, torch.ones_like(y), create_graph=True
    )[0]


# Network with exact temperature and fixed-end constraints
class ThermoElasticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(1, 40), nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
        )
        self.head_T = nn.Linear(40, 1)
        self.head_u = nn.Linear(40, 1)

    def forward(self, x):
        s = x / L
        features = self.shared(s)
        raw_T, raw_u = self.head_T(features), self.head_u(features)
        T = T0 + dT * (s + s * (1 - s) * raw_T)
        u = u_scale * s * raw_u
        return T, u


model = ThermoElasticNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=5000, gamma=0.5
)

x = torch.linspace(0, L, 100, device=device).reshape(-1, 1)
x.requires_grad_(True)
xL = torch.tensor([[L]], device=device, requires_grad=True)

# Training
print(f"Training on {device}...\n")
losses = []
best_loss = float("inf")
best_state = None
for epoch in range(15000):
    optimizer.zero_grad()
    x.grad = None
    xL.grad = None

    T, u = model(x)
    T_x, u_x = derivative(T, x), derivative(u, x)
    T_xx, u_xx = derivative(T_x, x), derivative(u_x, x)

    # Dimensionless PDE residuals
    r_heat = (k * T_xx) / (k * T_scale / L**2)
    r_mech = E * (u_xx - alpha * T_x) / (E * curvature_scale)
    loss_heat = torch.mean(r_heat**2)
    loss_mech = torch.mean(r_mech**2)

    # Remaining natural BC: sigma(L)=0
    T_L, u_L = model(xL)
    u_x_L = derivative(u_L, xL)
    sigma_L = E * (u_x_L - alpha * (T_L - T_ref))
    loss_stress = torch.mean((sigma_L / (E * strain_scale))**2)

    loss = loss_heat + loss_mech + 100 * loss_stress
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

    if (epoch + 1) % 5000 == 0:
        print(f"Epoch {epoch + 1}: Loss = {current_loss:.2e}")

# Evaluate the best iterate, not a possible final-epoch spike.
model.load_state_dict(best_state)
print(f"Best training loss: {best_loss:.2e}")

# Evaluation
x_test = torch.linspace(0, L, 200, device=device).reshape(-1, 1)
with torch.no_grad():
    T_pred, u_pred = model(x_test)

x_np = x_test.cpu().numpy().flatten()
T_pred = T_pred.cpu().numpy().flatten()
u_pred = u_pred.cpu().numpy().flatten()
T_true, u_true = exact_T(x_np), exact_u(x_np)

T_error = np.max(np.abs(T_pred - T_true))
u_error = np.max(np.abs(u_pred - u_true))
print("\n=== Results ===")
print(f"Maximum temperature error: {T_error:.6f} °C")
print(f"Maximum displacement error: {u_error * 1e6:.3f} μm")
print(f"Tip displacement (exact):   {u_true[-1] * 1e6:.3f} μm")
print(f"Tip displacement (PINN):    {u_pred[-1] * 1e6:.3f} μm")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(x_np, T_true, "b-", linewidth=2, label="Exact")
axes[0].plot(x_np, T_pred, "r--", linewidth=2, label="PINN")
axes[0].set(xlabel="x [m]", ylabel="T [°C]", title="Temperature Field")

axes[1].plot(x_np, u_true * 1e6, "b-", linewidth=2, label="Exact")
axes[1].plot(x_np, u_pred * 1e6, "r--", linewidth=2, label="PINN")
axes[1].set(xlabel="x [m]", ylabel="u [μm]", title="Thermal Expansion")

axes[2].semilogy(losses, color="darkgreen")
axes[2].set(xlabel="Epoch", ylabel="Loss", title="Training Convergence")

for axis in axes:
    axis.grid(True, alpha=0.3)
axes[0].legend()
axes[1].legend()
plt.tight_layout()
plt.savefig("thermoelastic.png", dpi=200, bbox_inches="tight")
plt.show()
