# Code Walkthrough: Autograd Verification

This document explains the implementation line by line. This script does NOT train a neural network - it proves that PyTorch's autograd computes exact derivatives, which is the foundation every PINN in this repository depends on.

---

## Step 1: Imports and Precision Setup

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

```

We use float64 (double precision) instead of the default float32. This matters because we want to see errors at the 10^-15 level (machine epsilon for 64-bit floats). With float32, machine epsilon is only 10^-7, so we couldn't distinguish "autograd is exact" from "autograd has small errors." The seed is for reproducibility but doesn't affect results here since no random initialization is involved.

---

## Step 2: The Derivative Helper Function

```python
def derivative(output, coordinate):
    return torch.autograd.grad(
        output,
        coordinate,
        torch.ones_like(output),
        create_graph=True,
    )[0]

```

This is the core utility that every PINN in the repository uses. It computes d(output)/d(coordinate) using reverse-mode automatic differentiation.

- `torch.ones_like(output)` is the seed gradient (since output has multiple elements, we need to specify how to combine them - ones gives element-wise derivatives)
- `create_graph=True` keeps the computational graph alive so we can differentiate again (needed for second derivatives like f_xx or u_xx)
- `[0]` extracts the tensor from the tuple that `autograd.grad` returns

This function is NOT finite differences. It walks the computational graph backwards, applying the chain rule exactly at each operation. The only error is floating-point roundoff.

---

## Step 3: Build the (x, t) Grid for Part 1

```python
x_values = torch.linspace(0, 1, 31)
t_values = torch.linspace(0, 1, 31)
X, T = torch.meshgrid(x_values, t_values, indexing="ij")
x = X.reshape(-1, 1).clone().detach().requires_grad_(True)
t = T.reshape(-1, 1).clone().detach().requires_grad_(True)

```

We create a 31x31 grid over [0,1] x [0,1] - that's 961 test points. The reshape to (-1, 1) flattens the grid into column vectors (each point becomes a row). `requires_grad_(True)` tells PyTorch to track operations on these tensors so autograd can differentiate with respect to them.

---

## Step 4: Compute Autograd Derivatives of f(x,t)

```python
f = torch.sin(torch.pi * x) * torch.exp(-t)
f_x = derivative(f, x)
f_t = derivative(f, t)
f_xx = derivative(f_x, x)

```

We pick f(x,t) = sin(pi*x) * exp(-t) because its derivatives are easy to compute by hand:

- df/dx = pi * cos(pi*x) * exp(-t)
- df/dt = -sin(pi*x) * exp(-t)
- d2f/dx2 = -pi^2 * sin(pi*x) * exp(-t)

The call `derivative(f_x, x)` computes the second derivative - this works because `create_graph=True` kept the graph from the first differentiation alive.

---

## Step 5: Compare to Analytical Derivatives

```python
f_x_exact = torch.pi * torch.cos(torch.pi * x) * torch.exp(-t)
f_t_exact = -torch.sin(torch.pi * x) * torch.exp(-t)
f_xx_exact = -(torch.pi**2) * torch.sin(torch.pi * x) * torch.exp(-t)

error_x = torch.abs(f_x - f_x_exact)
error_t = torch.abs(f_t - f_t_exact)
error_xx = torch.abs(f_xx - f_xx_exact)

```

We compute the same derivatives by hand (the formulas above) and take the absolute difference. If autograd is exact, these errors should be at machine epsilon (~10^-15 for float64). If we were using finite differences instead, errors would be around 10^-4 to 10^-8 depending on step size.

---

## Step 6: Build the (x, t) Grid for Part 2

```python
alpha = 0.01
x_heat_values = torch.linspace(0, 1, 101)
t_heat_values = torch.linspace(0, 0.5, 61)
X_heat, T_heat = torch.meshgrid(x_heat_values, t_heat_values, indexing="ij")
x_heat = X_heat.reshape(-1, 1).clone().detach().requires_grad_(True)
t_heat = T_heat.reshape(-1, 1).clone().detach().requires_grad_(True)

```

Part 2 uses a finer grid (101x61 = 6161 points) and a real PDE: the heat equation u_t = alpha * u_xx with alpha = 0.01. The domain is x in [0,1], t in [0, 0.5].

---

## Step 7: Verify the Heat Equation Residual

```python
u = torch.sin(torch.pi * x_heat) * torch.exp(
    -(torch.pi**2) * alpha * t_heat
)
u_t = derivative(u, t_heat)
u_x = derivative(u, x_heat)
u_xx = derivative(u_x, x_heat)
heat_residual = u_t - alpha * u_xx

```

This is the key test. We know that u(x,t) = sin(pi*x) * exp(-pi^2 * alpha * t) is an exact solution of the heat equation. You can verify by hand:

- u_t = -pi^2 * alpha * sin(pi*x) * exp(-pi^2*alpha*t)
- u_xx = -pi^2 * sin(pi*x) * exp(-pi^2*alpha*t)
- u_t - alpha * u_xx = -pi^2*alpha*u - alpha*(-pi^2*u) = 0

So the residual should be exactly zero. In practice it's ~10^-15 (roundoff). This is exactly what a PINN does during training - except the PINN uses a neural network for u instead of the exact formula, so its residual starts large and shrinks as training progresses.

---

## Step 8: Print Results

```python
print("--- Autograd versus analytical derivatives ---")
print(f"df/dx   maximum error: {error_x.max().item():.3e}")
print(f"df/dt   maximum error: {error_t.max().item():.3e}")
print(f"d2f/dx2 maximum error: {error_xx.max().item():.3e}")

print("\n--- Heat equation: u_t-alpha*u_xx=0 ---")
print(f"Mean absolute residual: {heat_residual.abs().mean().item():.3e}")
print(f"Maximum residual:       {heat_residual.abs().max().item():.3e}")

```

Expected output: all values around 10^-15 to 10^-16. This confirms autograd introduces no approximation error - only the unavoidable roundoff from 64-bit arithmetic.

---

## Step 9: Visualization

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

```

Three panels side by side:

**Panel 1** - The function f(x,t) itself. A contour plot showing sin(pi*x)*exp(-t) decaying over time. This is just context so the reader knows what we're differentiating.

**Panel 2** - log10 of the error in f_xx (autograd vs analytical). The colorbar should show values around -14 to -16 everywhere. If any region showed -4 or -5, that would indicate a problem.

**Panel 3** - log10 of the heat equation residual. Same idea - should be uniformly near -15 across the entire space-time domain, proving the PDE is satisfied to machine precision.

```python
log_error = np.log10(np.maximum(error_xx_grid, 1e-18))

```

The `np.maximum(..., 1e-18)` prevents log10(0) which would give -infinity. We cap at 10^-18 since anything below that is numerically meaningless.

```python
plt.savefig("chapter2_autograd_heat_equation.png", dpi=200, bbox_inches="tight")

```

Saves the 3-panel figure. The filename uses "chapter2" because this is the foundational chapter of the repository - it proves the tool before we use it.

---

## Training Flow

There is no training loop in this script. The "flow" is:

```
Define test function f(x,t)
        |
        v
Compute derivatives via autograd
        |
        v
Compute same derivatives by hand
        |
        v
Compare: error ~ 10^-15?  -----> YES: autograd is exact
        |
        v
Plug exact PDE solution into residual via autograd
        |
        v
Residual ~ 10^-15?  -----------> YES: PDE verification works
        |
        v
Conclusion: safe to use autograd for all PINN projects

```

---

## Connection to the Other Projects

Every other project in this repository does this:

1. Define a neural network u_NN(x)
2. Compute PDE residual using `derivative(u_NN, x)` (same function as here)
3. Minimize that residual during training

This script proves step 2 is exact. Any error in the trained PINN comes from the network's limited capacity or insufficient training - NOT from the differentiation tool.

| What this script proves | What it means for PINNs |
| --- | --- |
| df/dx via autograd matches analytical formula | First derivatives in momentum/heat equations are exact |
| d2f/dx2 via autograd matches analytical formula | Second derivatives (diffusion, elasticity) are exact |
| PDE residual of known solution is ~0 | If a PINN residual is large, the network is wrong - not autograd |

