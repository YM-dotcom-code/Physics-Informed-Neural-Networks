import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Homogeneous BVP: u''+u=0, u(0)=u(pi)=0
# General solution: u(x)=C*sin(x), so the amplitude C is undetermined.


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1),
        )

    def forward(self, s):
        return self.net(s)


def derivative(output, coordinate):
    return torch.autograd.grad(
        output,
        coordinate,
        torch.ones_like(output),
        create_graph=True,
    )[0]


def trial_solution(model, s, enforce_amplitude):
    boundary_factor = 4 * s * (1 - s)
    if enforce_amplitude:
        # Also enforces u(pi/2)=1 without prescribing the sine shape.
        return boundary_factor * (1 + (s - 0.5) * model(s))
    return boundary_factor * model(s)


def train_case(enforce_amplitude, epochs=5000):
    torch.manual_seed(SEED)
    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=2500, gamma=0.5
    )

    s = torch.linspace(0, 1, 60, device=device).reshape(-1, 1)
    s.requires_grad_(True)

    losses, best_losses = [], []
    best_loss = float("inf")
    best_state = None

    for _ in range(epochs):
        optimizer.zero_grad()
        s.grad = None

        u = trial_solution(model, s, enforce_amplitude)
        u_s = derivative(u, s)
        u_ss = derivative(u_s, s)
        residual = u_ss / torch.pi**2 + u
        loss = torch.mean(residual**2)

        loss.backward()
        optimizer.step()
        scheduler.step()

        current_loss = loss.item()
        losses.append(current_loss)
        if current_loss < best_loss:
            best_loss = current_loss
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
        best_losses.append(best_loss)

    model.load_state_dict(best_state)
    s_test = torch.linspace(0, 1, 300, device=device).reshape(-1, 1)
    with torch.no_grad():
        prediction = trial_solution(
            model, s_test, enforce_amplitude
        ).cpu().numpy().flatten()

    x = s_test.cpu().numpy().flatten() * np.pi
    reference = np.sin(x)
    return {
        "x": x,
        "prediction": prediction,
        "reference": reference,
        "losses": losses,
        "best_losses": best_losses,
        "best_loss": best_loss,
        "max_amplitude": np.max(np.abs(prediction)),
        "max_error": np.max(np.abs(prediction - reference)),
        "relative_l2": np.linalg.norm(prediction - reference)
        / np.linalg.norm(reference),
    }


print(f"Training non-unique and normalized cases on {device}...\n")
underdetermined = train_case(enforce_amplitude=False)
normalized = train_case(enforce_amplitude=True)

print("=== Underdetermined homogeneous BVP ===")
print(f"Best PDE loss:            {underdetermined['best_loss']:.3e}")
print(f"Learned maximum amplitude:{underdetermined['max_amplitude']:.3e}")
print("The near-zero solution is valid because the amplitude is unspecified.\n")

print("=== Added condition u(pi/2)=1 ===")
print(f"Best total loss:          {normalized['best_loss']:.3e}")
print(f"Maximum error vs sin(x):  {normalized['max_error']:.3e}")
print(f"Relative L2 error:        {normalized['relative_l2']:.3e}")

# Plot the failure mode and its resolution
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].semilogy(
    underdetermined["best_losses"], "r-", label="Underdetermined"
)
axes[0].semilogy(normalized["best_losses"], "g-", label="With u(pi/2)=1")
axes[0].set(xlabel="Epoch", ylabel="Best loss", title="Training Convergence")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(
    underdetermined["x"], underdetermined["reference"],
    "b-", linewidth=2.5, label="Reference C=1",
)
axes[1].plot(
    underdetermined["x"], underdetermined["prediction"],
    "r--", linewidth=2, label="PINN",
)
axes[1].set(
    xlabel="x", ylabel="u(x)",
    title="Non-unique BVP: PINN Selects C≈0",
)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(
    normalized["x"], normalized["reference"],
    "b-", linewidth=2.5, label="Exact: sin(x)",
)
axes[2].plot(
    normalized["x"], normalized["prediction"],
    "g--", linewidth=2, label="PINN",
)
axes[2].set(
    xlabel="x", ylabel="u(x)",
    title="Unique BVP After Amplitude Condition",
)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "chapter1_trivial_solution_demo.png",
    dpi=200,
    bbox_inches="tight",
)
plt.show()
