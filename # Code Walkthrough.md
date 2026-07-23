# Code Walkthrough

This section explains the implementation line by line, mapping each block of code to the physics and PINN concepts described above.

---

## Step 1: Imports and Device Setup

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

```

PyTorch handles the neural network and automatic differentiation. NumPy and Matplotlib handle post-processing and visualization. The random seed ensures reproducibility. If a GPU is available, all tensors and computations are moved there for faster training.

---

## Step 2: Physical Parameters

```python
k = 50.0       # Thermal conductivity [W/mK]
E_mod = 200e9  # Young's modulus [Pa]
alpha = 12e-6  # Thermal expansion coefficient [1/K]
L = 1.0        # Bar length [m]
T0, TL = 100.0, 500.0  # Temperature BCs
T_ref = 100.0  # Reference temperature (stress-free)

```

These define the material properties and boundary conditions for a steel bar. The reference temperature is the temperature at which no thermal strain exists. All values use SI units.

---

## Step 3: Analytical Solutions

```python
def exact_T(x):
    return T0 + (TL - T0) * x / L

def exact_u(x):
    return alpha * (TL - T0) / (2 * L) * x**2

```

Because the governing equations have closed-form solutions for this problem, we define them here for validation. The temperature is linear (steady conduction with no source). The displacement is quadratic (thermal strain integrated with zero stress everywhere).

The derivation:

- From `dσ/dx = 0` and `σ(L) = 0`, stress is zero throughout the bar.
- Therefore `du/dx = α(T - T_ref) = α(TL - T0)x/L`.
- Integrating with `u(0) = 0` gives the parabolic displacement field.

---

## Step 4: Neural Network Architecture

```python
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
        h = self.shared(x)
        return self.head_T(h), self.head_u(h)

```

The network takes a single input (the spatial coordinate `x`) and produces two outputs: temperature `T(x)` and displacement `u(x)`.

The architecture uses a **shared trunk** with 4 hidden layers of 40 neurons each, followed by two separate **output heads**. The shared layers learn spatial features common to both physical fields. The separate heads allow each field to specialize its final mapping.

The `tanh` activation function is chosen because it is infinitely differentiable, which matters since we compute second derivatives through autograd during training.

---

## Step 5: Optimizer and Scheduler

```python
model = ThermoElasticNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

```

Adam is the standard optimizer for PINNs. The learning rate starts at `1e-3` and is halved every 5000 epochs. This schedule allows aggressive exploration early in training, then fine-tuning as the solution converges.

---

## Step 6: Collocation Points

```python
n_int = 100
x_int = torch.linspace(0, L, n_int, device=device).reshape(-1, 1)
x_int.requires_grad_(True)
x0 = torch.tensor([[0.0]], device=device, requires_grad=True)
xL_t = torch.tensor([[L]], device=device, requires_grad=True)

```

100 uniformly spaced points span the domain `[0, 1]`. These are the collocation points where the PDE residual will be evaluated. `requires_grad_(True)` enables PyTorch to compute derivatives of the network output with respect to these spatial coordinates.

Two additional points (`x0` and `xL_t`) are created for evaluating boundary conditions at the left and right ends.

---

## Step 7: Training Loop — PDE Residuals

```python
T, u = model(x_int)

# Heat equation: k*T'' = 0
T_x = torch.autograd.grad(T, x_int, torch.ones_like(T), create_graph=True)[0]
T_xx = torch.autograd.grad(T_x, x_int, torch.ones_like(T_x), create_graph=True)[0]
res_heat = k * T_xx
loss_heat = torch.mean(res_heat**2)

# Elasticity: E*u'' - E*alpha*T' = 0
u_x = torch.autograd.grad(u, x_int, torch.ones_like(u), create_graph=True)[0]
u_xx = torch.autograd.grad(u_x, x_int, torch.ones_like(u_x), create_graph=True)[0]
res_elast = E_mod * u_xx - E_mod * alpha * T_x
loss_elast = torch.mean(res_elast**2) / E_mod**2

```

This is the core of the PINN. At each collocation point:

1. The network predicts `T` and `u`.
2. `torch.autograd.grad` computes exact first and second derivatives with respect to `x`.
3. These derivatives are substituted into the governing PDEs to compute the **residual** — how much the current prediction violates the physics.
4. The mean squared residual becomes the physics loss.

The `create_graph=True` flag is critical: it keeps the computational graph alive so that gradients can flow through the derivative computation during backpropagation.

The elasticity residual is normalized by `E_mod²` to bring it to a comparable scale with the heat residual, since Young's modulus is `200 × 10⁹`.

**Note on the coupling**: The temperature gradient `T_x` appears in the elasticity residual. This is how the thermal field drives the mechanical response — the network must learn a displacement field whose second derivative matches the thermal gradient.

---

## Step 8: Training Loop — Boundary Conditions

```python
T_0, u_0 = model(x0)
T_L, u_L = model(xL_t)

# Temperature BCs: T(0)=100, T(L)=500
loss_T_bc = (T_0 - T0)**2 + (T_L - TL)**2

# Displacement BC: u(0)=0
loss_u_bc = u_0**2

# Stress-free BC at x=L: du/dx(L) = alpha*(T(L) - T_ref)
u_L_val = model(xL_t)[1]
u_x_L = torch.autograd.grad(u_L_val, xL_t, torch.ones_like(u_L_val), create_graph=True)[0]
target_strain = alpha * (TL - T_ref)
loss_stress_bc = (u_x_L - target_strain)**2

```

Boundary conditions are enforced as soft penalties:

- **Dirichlet BCs for temperature**: The network output at `x=0` and `x=L` must match the prescribed temperatures (100°C and 500°C).
- **Dirichlet BC for displacement**: The fixed end at `x=0` requires zero displacement.
- **Neumann BC for stress**: The free end at `x=L` requires zero stress, which translates to `du/dx(L) = α(T(L) - T_ref)`. This is the strain that produces zero mechanical stress given the local thermal expansion.

---

## Step 9: Loss Assembly and Weight Update

```python
loss_bc = loss_T_bc.squeeze() + 1e12*loss_u_bc.squeeze() + 1e12*loss_stress_bc.squeeze()
loss = loss_heat + loss_elast + 100*loss_bc

loss.backward()
optimizer.step()
scheduler.step()

```

The total loss combines physics residuals and boundary condition penalties. The weights reflect the relative importance and scale of each term:

- `1e12` on displacement and stress BCs: These values are extremely small in magnitude (micrometers and microstrain), so large multipliers bring their loss contribution to a meaningful scale relative to the heat equation.
- `100` on the combined BC loss: Prioritizes boundary satisfaction over interior PDE accuracy during early training.

`loss.backward()` computes gradients of the total loss with respect to all network weights via backpropagation. `optimizer.step()` updates the weights. The scheduler reduces the learning rate at the scheduled intervals.

---

## Step 10: Evaluation and Plotting

```python
x_test = torch.linspace(0, L, 200, device=device).reshape(-1, 1)
with torch.no_grad():
    T_pred, u_pred = model(x_test)

```

After training, the network is evaluated on a denser grid (200 points) with gradient computation disabled (`torch.no_grad()`). The predictions are compared against the analytical solutions to quantify accuracy.

The three-panel plot shows:

1. **Temperature field** — PINN vs exact (should overlap perfectly for this linear problem)
2. **Displacement field** — PINN vs exact (quadratic curve, max 2400 μm at free end)
3. **Training convergence** — loss vs epoch on a log scale

---

## Summary of the Training Flow

```text
┌─────────────────────────────────────────────────────────────────┐
│                         Each Epoch                                │
│                                                                   │
│   x_int ──→ model(x_int) ──→ T, u                               │
│                                  │                                │
│                    ┌─────────────┼─────────────┐                 │
│                    ▼             ▼             ▼                  │
│              autograd(T)   autograd(u)   model(x0, xL)           │
│              T', T''       u', u''       T_0, T_L, u_0           │
│                    │             │             │                  │
│                    ▼             ▽             ▼                  │
│              k·T'' = 0    E·u''-E·α·T'=0   BC penalties          │
│              (residual)    (residual)     (Dirichlet+Neumann)    │
│                    │             │             │                  │
│                    └──────┬──────┘─────────────┘                 │
│                           ▼                                       │
│                      Total Loss                                   │
│                           │                                       │
│                           ▼                                       │
│                    loss.backward()                                │
│                    optimizer.step()                               │
│                                                                   │
│   Repeat × 15,000 epochs                                         │
└─────────────────────────────────────────────────────────────────┘

```

