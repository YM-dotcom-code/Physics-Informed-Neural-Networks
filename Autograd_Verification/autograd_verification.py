import torch
import numpy as np
import matplotlib.pyplot as plt


# Float64 makes floating-point differentiation errors easier to assess.
torch.set_default_dtype(torch.float64)
torch.manual_seed(42)


def derivative(output, coordinate):
    return torch.autograd.grad(
        output,
        coordinate,
        torch.ones_like(output),
        create_graph=True,
    )[0]


print("=== AUTOMATIC DIFFERENTIATION FOR PINNs ===")
print("Autograd differentiates the computational graph up to roundoff.\n")

# Part 1: verify derivatives of f(x,t)=sin(pi*x)*exp(-t)
x_values = torch.linspace(0, 1, 31)
t_values = torch.linspace(0, 1, 31)
X, T = torch.meshgrid(x_values, t_values, indexing="ij")
x = X.reshape(-1, 1).clone().detach().requires_grad_(True)
t = T.reshape(-1, 1).clone().detach().requires_grad_(True)

f = torch.sin(torch.pi * x) * torch.exp(-t)
f_x = derivative(f, x)
f_t = derivative(f, t)
f_xx = derivative(f_x, x)

f_x_exact = torch.pi * torch.cos(torch.pi * x) * torch.exp(-t)
f_t_exact = -torch.sin(torch.pi * x) * torch.exp(-t)
f_xx_exact = -(torch.pi**2) * torch.sin(torch.pi * x) * torch.exp(-t)

error_x = torch.abs(f_x - f_x_exact)
error_t = torch.abs(f_t - f_t_exact)
error_xx = torch.abs(f_xx - f_xx_exact)

print("--- Autograd versus analytical derivatives ---")
print(f"df/dx   maximum error: {error_x.max().item():.3e}")
print(f"df/dt   maximum error: {error_t.max().item():.3e}")
print(f"d²f/dx² maximum error: {error_xx.max().item():.3e}")


# Part 2: verify the heat-equation residual over a full space-time grid
alpha = 0.01
x_heat_values = torch.linspace(0, 1, 101)
t_heat_values = torch.linspace(0, 0.5, 61)
X_heat, T_heat = torch.meshgrid(
    x_heat_values, t_heat_values, indexing="ij"
)
x_heat = X_heat.reshape(-1, 1).clone().detach().requires_grad_(True)
t_heat = T_heat.reshape(-1, 1).clone().detach().requires_grad_(True)

u = torch.sin(torch.pi * x_heat) * torch.exp(
    -(torch.pi**2) * alpha * t_heat
)
u_t = derivative(u, t_heat)
u_x = derivative(u, x_heat)
u_xx = derivative(u_x, x_heat)
heat_residual = u_t - alpha * u_xx

print("\n--- Heat equation: u_t-alpha*u_xx=0 ---")
print(f"Mean absolute residual: {heat_residual.abs().mean().item():.3e}")
print(f"Maximum residual:       {heat_residual.abs().max().item():.3e}")
print("Residuals are limited by floating-point roundoff.")
print("This verifies PINN differentiation; no neural network is trained here.")

# Visualization
shape_f = (len(x_values), len(t_values))
shape_heat = (len(x_heat_values), len(t_heat_values))
f_grid = f.detach().cpu().numpy().reshape(shape_f)
error_xx_grid = error_xx.detach().cpu().numpy().reshape(shape_f)
u_grid = u.detach().cpu().numpy().reshape(shape_heat)
residual_grid = heat_residual.detach().cpu().numpy().reshape(shape_heat)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

plot1 = axes[0].contourf(
    T.cpu().numpy(), X.cpu().numpy(), f_grid, levels=30, cmap="viridis"
)
axes[0].set(xlabel="t", ylabel="x", title="f(x,t)=sin(pi*x) exp(-t)")
fig.colorbar(plot1, ax=axes[0], label="f")

log_error = np.log10(np.maximum(error_xx_grid, 1e-18))
plot2 = axes[1].contourf(
    T.cpu().numpy(), X.cpu().numpy(), log_error, levels=30, cmap="magma"
)
axes[1].set(
    xlabel="t", ylabel="x",
    title="log10 Error in Autograd f_xx",
)
fig.colorbar(plot2, ax=axes[1], label="log10 absolute error")

log_residual = np.log10(np.maximum(np.abs(residual_grid), 1e-18))
plot3 = axes[2].contourf(
    T_heat.cpu().numpy(), X_heat.cpu().numpy(),
    log_residual, levels=30, cmap="magma",
)
axes[2].set(
    xlabel="t", ylabel="x",
    title="log10 Heat-Equation Residual",
)
fig.colorbar(plot3, ax=axes[2], label="log10 absolute residual")

plt.tight_layout()
plt.savefig(
    "chapter2_autograd_heat_equation.png",
    dpi=200,
    bbox_inches="tight",
)
plt.show()
