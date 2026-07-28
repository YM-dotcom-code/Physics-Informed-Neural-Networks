import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inverse steady heat problem: k*T''+Q=0
k_true = 50.0
k_initial = 10.0
Q = 5000.0
L = 1.0
T_left = T_right = 20.0
noise_std = 0.1


def exact_temperature(x, conductivity):
    return (
        -Q * x**2 / (2 * conductivity)
        + Q * L * x / (2 * conductivity)
        + T_left
    )


# Synthetic measurements
x_sensors = np.linspace(0.1, 0.9, 10)
T_sensors = exact_temperature(x_sensors, k_true)
T_sensors += np.random.normal(0, noise_std, len(x_sensors))
T_scale = max(np.max(np.abs(T_sensors - T_left)), 1.0)

s_data = torch.tensor(
    x_sensors / L, dtype=torch.float32, device=device
).reshape(-1, 1)
T_data = torch.tensor(
    T_sensors, dtype=torch.float32, device=device
).reshape(-1, 1)


def derivative(output, coordinate):
    return torch.autograd.grad(
        output,
        coordinate,
        torch.ones_like(output),
        create_graph=True,
    )[0]


class InverseHeatNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 30), nn.Tanh(),
            nn.Linear(30, 30), nn.Tanh(),
            nn.Linear(30, 30), nn.Tanh(),
            nn.Linear(30, 1),
        )
        # Log parameter guarantees k>0 and improves multiplicative updates.
        self.log_k = nn.Parameter(
            torch.tensor(np.log(k_initial), dtype=torch.float32, device=device)
        )

    @property
    def conductivity(self):
        return torch.exp(self.log_k)

    def forward(self, s):
        raw_temperature = self.net(s)
        baseline = T_left * (1 - s) + T_right * s
        return baseline + T_scale * s * (1 - s) * raw_temperature


model = InverseHeatNet().to(device)
network_parameters = [
    parameter
    for name, parameter in model.named_parameters()
    if name != "log_k"
]
optimizer = torch.optim.Adam(
    [
        {"params": network_parameters, "lr": 1e-3},
        {"params": [model.log_k], "lr": 5e-3},
    ]
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=5000, gamma=0.5
)

s = torch.linspace(0, 1, 80, device=device).reshape(-1, 1)
s.requires_grad_(True)

losses, best_losses, k_history = [], [], []
best_loss = float("inf")
best_state = None
print(f"Training inverse conductivity PINN on {device}...\n")

for epoch in range(10000):
    optimizer.zero_grad()
    s.grad = None

    temperature = model(s)
    T_s = derivative(temperature, s)
    T_ss = derivative(T_s, s)

    # T_xx=T_ss/L²; divide the physical residual by Q.
    residual = (
        model.conductivity * T_ss / L**2 + Q
    ) / Q
    loss_pde = torch.mean(residual**2)

    predicted_data = model(s_data)
    loss_data = torch.mean(
        ((predicted_data - T_data) / T_scale) ** 2
    )
    loss = loss_pde + 100 * loss_data

    loss.backward()
    optimizer.step()
    scheduler.step()

    current_loss = loss.item()
    losses.append(current_loss)
    k_history.append(model.conductivity.item())
    if current_loss < best_loss:
        best_loss = current_loss
        best_state = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
    best_losses.append(best_loss)

    if (epoch + 1) % 2000 == 0:
        print(
            f"Epoch {epoch + 1}: "
            f"k={model.conductivity.item():.3f} W/(m*K), "
            f"loss={current_loss:.2e}"
        )

model.load_state_dict(best_state)
k_predicted = model.conductivity.item()
k_error = abs(k_predicted - k_true) / k_true * 100
print(f"Best training loss: {best_loss:.2e}")

# Evaluation
s_test = torch.linspace(0, 1, 200, device=device).reshape(-1, 1)
s_test.requires_grad_(True)
T_pred_tensor = model(s_test)
T_s = derivative(T_pred_tensor, s_test)
T_ss = derivative(T_s, s_test)
physical_residual = (
    model.conductivity * T_ss / L**2 + Q
).detach().cpu().numpy().flatten()

x = s_test.detach().cpu().numpy().flatten() * L
T_pred = T_pred_tensor.detach().cpu().numpy().flatten()
T_exact = exact_temperature(x, k_true)
temperature_error = np.max(np.abs(T_pred - T_exact))
sensor_rmse = np.sqrt(
    np.mean((model(s_data).detach().cpu().numpy().flatten() - T_sensors) ** 2)
)

print("\n=== Inverse problem results ===")
print(f"True conductivity:      {k_true:.4f} W/(m*K)")
print(f"Predicted conductivity: {k_predicted:.4f} W/(m*K)")
print(f"Conductivity error:     {k_error:.4f}%")
print(f"Maximum T error:        {temperature_error:.4f} °C")
print(f"Sensor RMSE:            {sensor_rmse:.4f} °C")
print(f"Mean |PDE residual|:    {np.mean(np.abs(physical_residual)):.3e}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].plot(x, T_exact, "b-", linewidth=2.5, label="Exact")
axes[0].plot(x, T_pred, "r--", linewidth=2, label="PINN")
axes[0].scatter(
    x_sensors, T_sensors, color="green", s=45, zorder=5,
    label="Noisy sensors",
)
axes[0].set(xlabel="x [m]", ylabel="T [°C]", title="Temperature Field")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(k_history, color="navy", linewidth=1.5, label="Learned k")
axes[1].axhline(k_true, color="red", linestyle="--", label="True k")
axes[1].set(
    xlabel="Epoch", ylabel="k [W/(m*K)]",
    title="Conductivity Identification",
)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].semilogy(losses, color="gray", alpha=0.4, label="Training loss")
axes[2].semilogy(best_losses, color="darkgreen", label="Best loss")
axes[2].set(xlabel="Epoch", ylabel="Normalized loss", title="Convergence")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("chapter8_inverse_problem.png", dpi=200, bbox_inches="tight")
plt.show()
