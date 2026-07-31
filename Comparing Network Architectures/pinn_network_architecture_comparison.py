import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
EPOCHS = 5000

# u''(x)=-sin(x), x in [0,pi], u(0)=u(pi)=0
# Exact solution: u(x)=sin(x)


class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class PINN_A(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1),
        )

    def forward(self, s):
        return self.net(s)


class PINN_B(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1),
        )

    def forward(self, s):
        return self.net(s)


class PINN_C(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20), SinActivation(),
            nn.Linear(20, 20), SinActivation(),
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


def trial_solution(model, s):
    # Exact homogeneous BCs at s=0 and s=1.
    return 4 * s * (1 - s) * model(s)


def train_model(model_class, name):
    torch.manual_seed(SEED)
    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=2500, gamma=0.5
    )
    s = torch.linspace(0, 1, 60, device=device).reshape(-1, 1)
    s.requires_grad_(True)

    losses, best_losses = [], []
    best_loss = float("inf")
    best_state = None
    convergence_epoch = None
    start = time.perf_counter()

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        s.grad = None

        u = trial_solution(model, s)
        u_s = derivative(u, s)
        u_ss = derivative(u_s, s)
        # u_xx=u_ss/pi² and x=pi*s.
        residual = u_ss / torch.pi**2 + torch.sin(torch.pi * s)
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

        if convergence_epoch is None and best_loss < 1e-5:
            convergence_epoch = epoch + 1

    runtime = time.perf_counter() - start
    model.load_state_dict(best_state)

    s_test = torch.linspace(0, 1, 300, device=device).reshape(-1, 1)
    with torch.no_grad():
        prediction = trial_solution(model, s_test).cpu().numpy().flatten()
    x = s_test.cpu().numpy().flatten() * np.pi
    exact = np.sin(x)
    max_error = np.max(np.abs(prediction - exact))
    relative_l2 = np.linalg.norm(prediction - exact) / np.linalg.norm(exact)
    parameter_count = sum(p.numel() for p in model.parameters())

    return {
        "name": name,
        "x": x,
        "prediction": prediction,
        "exact": exact,
        "losses": losses,
        "best_losses": best_losses,
        "best_loss": best_loss,
        "max_error": max_error,
        "relative_l2": relative_l2,
        "convergence_epoch": convergence_epoch,
        "parameters": parameter_count,
        "runtime": runtime,
    }


architectures = [
    (PINN_A, "Small tanh [1,20,20,1]"),
    (PINN_B, "Wide/deep tanh [1,50,50,50,1]"),
    (PINN_C, "Small sine [1,20,20,1]"),
]

print(f"Comparing PINN architectures on {device}...\n")
results = [train_model(model_class, name) for model_class, name in architectures]

for result in results:
    convergence = result["convergence_epoch"]
    convergence_text = str(convergence) if convergence is not None else "not reached"
    print(
        f"{result['name']}: parameters={result['parameters']}, "
        f"max error={result['max_error']:.3e}, "
        f"relative L2={result['relative_l2']:.3e}, "
        f"loss<1e-5 at epoch={convergence_text}, "
        f"runtime={result['runtime']:.1f}s"
    )

# Plot predictions and convergence
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(
    results[0]["x"], results[0]["exact"], "k-", linewidth=3,
    label="Exact: sin(x)",
)
colors = ["blue", "red", "green"]
for result, color in zip(results, colors):
    axes[0].plot(
        result["x"], result["prediction"], "--",
        color=color, linewidth=1.8, label=result["name"],
    )
    axes[1].semilogy(
        result["best_losses"], color=color, linewidth=1.8,
        label=result["name"],
    )

axes[0].set(xlabel="x", ylabel="u(x)", title="PINN Predictions")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)
axes[1].set(
    xlabel="Epoch", ylabel="Best PDE loss",
    title="Architecture Convergence",
)
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    "chapter3_architecture_comparison.png",
    dpi=200,
    bbox_inches="tight",
)
plt.show()
