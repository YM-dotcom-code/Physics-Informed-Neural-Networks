# CODE_WALKTHROUGH.md — Stokes Flow PINN

A line-by-line explanation of the pressure-driven channel flow PINN implementation.

---

## 1. Imports & Device Setup

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

```

- **torch**: Automatic differentiation (autograd) — the core of PINN training
- **torch.nn**: Neural network building blocks (layers, activations)
- **numpy/matplotlib**: Post-processing and visualization
- **Seeds**: Fixed at 42 for reproducible results
- **Device**: Uses GPU if available, falls back to CPU

---

## 2. Physical Parameters

```python
mu = 1.0
pressure_gradient = -2.0
h = 1.0

```

| Parameter | Meaning | Value |
| --- | --- | --- |
| `mu` | Dynamic viscosity (μ) | 1.0 Pa·s |
| `pressure_gradient` | dp/dx (negative = flow in +x direction) | −2.0 Pa/m |
| `h` | Half-channel height | 1.0 m |

The **negative** pressure gradient means pressure decreases in the flow direction (high pressure on the left pushes fluid to the right).

---

## 3. Characteristic Scales

```python
velocity_scale = max(
    abs(-pressure_gradient * h**2 / (2 * mu)), 1e-12
)
residual_scale = max(abs(pressure_gradient / mu), 1e-12)

```

| Scale | Formula | Value | Purpose |
| --- | --- | --- | --- |
| `velocity_scale` | |(-dp/dx)·h²/(2μ)| | 1.0 | Scales network output to physical units |
| `residual_scale` | |(dp/dx)/μ| | 2.0 | Normalizes PDE residual to O(1) |

**Why scale?**

- `velocity_scale` ensures the raw network output (which lives near [-1, 1]) maps to the correct physical magnitude
- `residual_scale` prevents the loss from being artificially large/small due to units
- The `max(..., 1e-12)` guards against division by zero if parameters are changed to zero

---

## 4. Analytical Solution

```python
def exact_velocity(y):
    return pressure_gradient * (y**2 - h**2) / (2 * mu)

```

**Derivation:**

```
μ·u'' = dp/dx
u'' = dp/dx / μ = -2
u' = -2y + C₁
u = -y² + C₁y + C₂

Apply u(-1) = 0 and u(+1) = 0:
→ C₁ = 0, C₂ = 1
→ u(y) = 1 - y²

```

**Verification with the code formula:**

```
pressure_gradient * (y² - h²) / (2*mu)
= -2 * (y² - 1) / (2*1)
= -(y² - 1)
= 1 - y²  ✓

```

Key values:

- `u(0) = 1.0 m/s` (centerline maximum)
- `u(±1) = 0` (walls)
- `u(±0.5) = 0.75 m/s`

---

## 5. Derivative Helper Function

```python
def derivative(output, coordinate):
    return torch.autograd.grad(
        output,
        coordinate,
        torch.ones_like(output),
        create_graph=True,
    )[0]

```

Uses PyTorch's automatic differentiation to compute **exact** derivatives:

- `output`: the tensor to differentiate (e.g., velocity u)
- `coordinate`: differentiate with respect to this (e.g., position y)
- `torch.ones_like(output)`: gradient seed (needed for batch differentiation)
- `create_graph=True`: allows computing second derivatives (u″) by calling this twice

**Usage pattern:**

```python
u_y = derivative(u, y)      # first derivative: du/dy
u_yy = derivative(u_y, y)   # second derivative: d²u/dy²

```

---

## 6. Neural Network Architecture

```python
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
        return velocity_scale * (1 - s**2) * raw_velocity

```

### Architecture Breakdown

```
y (position)
│
├─ Normalize: s = y / h          Maps [-1, +1] → [-1, +1]
│
├─ Linear(1 → 30) + Tanh         Input layer
├─ Linear(30 → 30) + Tanh        Hidden layer 1
├─ Linear(30 → 30) + Tanh        Hidden layer 2
├─ Linear(30 → 1)                Output layer → raw_velocity
│
└─ Hard constraint:
   u = velocity_scale × (1 - s²) × raw_velocity

```

### Why Tanh?

- Smooth and infinitely differentiable (C∞) — needed for computing u″ via autograd
- Outputs in [-1, 1] — prevents exploding values in early training
- Symmetric around zero — matches the symmetric physics (profile is symmetric about y=0)

### Input Normalization: s = y/h

- Maps the physical domain [−h, +h] = [−1, +1] to the normalized range [−1, +1]
- Neural networks work best when inputs are near zero
- With h = 1, this is technically a no-op, but it makes the code general for any h

### Hard Boundary Constraint

```python
return velocity_scale * (1 - s**2) * raw_velocity

```

The factor `(1 - s²)`:

- At s = −1 (bottom wall): (1−1) = 0 → **u = 0** always
- At s = +1 (top wall): (1−1) = 0 → **u = 0** always
- At s = 0 (center): (1−0) = 1 → maximum freedom

The network only needs to learn `raw_velocity(s)` such that the overall product matches the physics. Since the exact solution is `u = 1−s²`, the ideal `raw_velocity = 1/velocity_scale = 1.0` (a constant). This makes convergence extremely fast.

---

## 7. Optimizer, Scheduler, and Best-State Tracking

```python
model = StokesNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=5000, gamma=0.5
)

```

| Component | Setting | Purpose |
| --- | --- | --- |
| Optimizer | Adam, lr=1e-3 | Adaptive step sizes per parameter |
| Scheduler | StepLR, halve every 5000 | Big steps early → fine-tune later |
| Best-state | Save when loss improves | Avoid late-training noise/oscillations |

**Learning rate schedule:**

- Epochs 0–4999: lr = 0.001
- Epochs 5000–9999: lr = 0.0005

---

## 8. Collocation Points

```python
y = torch.linspace(-h, h, 80, device=device).reshape(-1, 1)
y.requires_grad_(True)

```

- **80 points** uniformly spaced in [−1, +1]
- `requires_grad_(True)`: enables autograd to compute derivatives through these points
- Shape `(-1, 1)`: column vector, one point per row (batch dimension × feature dimension)

**Why 80 is enough:** The exact solution is a simple parabola (degree-2 polynomial). Even 20 points would likely suffice. 80 is conservative.

---

## 9. Training Loop

```python
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

```

### Step-by-step each epoch:

| Step | Code | What happens |
| --- | --- | --- |
| 1 | `optimizer.zero_grad()` | Clear old gradients |
| 2 | `model(y)` | Network predicts velocity at 80 points |
| 3 | `derivative(velocity, y)` | Autograd computes u′(y) |
| 4 | `derivative(velocity_y, y)` | Autograd computes u″(y) |
| 5 | `velocity_yy - pressure_gradient/mu` | PDE residual (should be 0) |
| 6 | `mean((residual/scale)²)` | Normalized MSE loss |
| 7 | `loss.backward()` | Backpropagation |
| 8 | `optimizer.step()` | Update network weights |
| 9 | `scheduler.step()` | Check if lr should decrease |
| 10 | `if < best: save` | Track best model |

### The PDE Residual

```python
residual = velocity_yy - pressure_gradient / mu

```

The governing equation is `μ·u″ = dp/dx`, which rearranges to:

```
u″ − (dp/dx)/μ = 0

```

So the residual is `u″ − (dp/dx)/μ`. If the network's prediction is correct, this equals zero everywhere. The loss is the mean squared (normalized) residual.

### No Boundary Loss!

Unlike the thermoelastic project (which needed a soft BC for stress), this problem has **zero boundary loss terms**. Both BCs are hard-coded into the network output. The loss is purely:

```
loss = mean( (PDE residual / scale)² )

```

This makes training simpler — no loss balancing, no penalty weights to tune.

---

## 10. Post-Training: Load Best Model

```python
model.load_state_dict(best_state)

```

After all 10,000 epochs, we don't use the final model — we load the **best** model seen during training. This avoids issues where the loss might oscillate slightly in late epochs due to learning rate noise.

---

## 11. Evaluation

```python
y_test = torch.linspace(-h, h, 200, device=device).reshape(-1, 1)
with torch.no_grad():
    velocity_pred = model(y_test).cpu().numpy().flatten()

y_np = y_test.cpu().numpy().flatten()
velocity_exact = exact_velocity(y_np)
max_error = np.max(np.abs(velocity_pred - velocity_exact))
relative_l2 = np.linalg.norm(velocity_pred - velocity_exact) / np.linalg.norm(velocity_exact)
wall_error = max(abs(velocity_pred[0]), abs(velocity_pred[-1]))

```

| Metric | What it measures |
| --- | --- |
| `max_error` | Worst-case absolute error at any point |
| `relative_l2` | Overall error relative to solution magnitude |
| `wall_error` | Error at boundaries (should be machine zero) |
| Centerline velocity | u(0) — should be exactly 1.0 |

**Key insight:** `wall_error` is ~10⁻¹⁶ (machine epsilon) because the hard constraint makes it mathematically impossible for the velocity to be nonzero at the walls. This is NOT learned — it's guaranteed by the formula.

---

## 12. Visualization

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

```

**Left panel:** Velocity profile

- Blue solid line: exact analytical solution u(y) = 1 − y²
- Red dashed line: PINN prediction
- If training worked, these should be indistinguishable

**Right panel:** Training convergence

- Gray: raw training loss per epoch
- Dark green: best loss seen so far (monotonically decreasing)
- Should drop several orders of magnitude over 10,000 epochs

---

## Training Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│                     INITIALIZATION                       │
├─────────────────────────────────────────────────────────┤
│  • Create StokesNet (3×30, tanh, hard BCs)              │
│  • Adam optimizer (lr = 1e-3)                           │
│  • 80 collocation points in [-1, +1]                    │
│  • StepLR scheduler (halve every 5000)                  │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              TRAINING LOOP (×10,000)                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐     ┌───────────────┐                 │
│  │ y (80 pts)  │────▶│  StokesNet    │────▶ u(y)       │
│  └─────────────┘     └───────────────┘                 │
│                              │                          │
│                              ▼                          │
│                     ┌─────────────────┐                 │
│                     │  Autograd:      │                 │
│                     │  u′ = du/dy     │                 │
│                     │  u″ = d²u/dy²  │                 │
│                     └────────┬────────┘                 │
│                              │                          │
│                              ▼                          │
│         residual = u″ − (dp/dx)/μ                      │
│         loss = mean( (residual / scale)² )              │
│                              │                          │
│                              ▼                          │
│         loss.backward() → optimizer.step()              │
│         if loss < best → save state                     │
│                                                         │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    EVALUATION                            │
├─────────────────────────────────────────────────────────┤
│  • Load best model state                                │
│  • Predict on 200 test points                           │
│  • Compare to exact solution                            │
│  • Plot velocity profile + convergence                  │
└─────────────────────────────────────────────────────────┘

```

---

## Key Differences From Thermoelastic PINN

| Aspect | Thermoelastic Bar | Channel Flow |
| --- | --- | --- |
| Physics | 2 coupled PDEs | 1 single ODE |
| Outputs | 2 (T and u) | 1 (velocity) |
| Architecture | 4×64, two heads | 3×30, single output |
| Hard BCs | 3 (T₀, T_L, u₀) | 2 (both walls) |
| Soft BCs | 1 (stress-free σ(L)=0) | 0 (none!) |
| Loss terms | 3 (heat + mech + BC) | 1 (PDE only) |
| Balancing weights | 100 on BC | None needed |
| Difficulty | Moderate (coupling + mixed BCs) | Easy (single smooth ODE) |
| Expected accuracy | ~10⁻⁴ | ~10⁻⁶ |

