# Code Walkthrough

This document explains the implementation line by line, mapping each block of code to the physics and PINN concepts described in the README.

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

PyTorch handles the neural network and automatic differentiation. NumPy and Matplotlib handle post-processing and visualization. The random seed ensures reproducibility — run the same script twice, get the same model. If a GPU is available, all tensors and computations are moved there for faster training.

---

## Step 2: Physical Parameters

```python
k = 50.0        # Thermal conductivity [W/mK]
E_mod = 200e9   # Young's modulus [Pa]
alpha = 12e-6   # Thermal expansion coefficient [1/K]
L = 1.0         # Bar length [m]
T0, TL = 100.0, 500.0   # Temperature boundary conditions [°C]
T_ref = T0               # Reference temperature (stress-free state)
dT = TL - T0             # Temperature difference across bar = 400°C
```

These define the material properties and boundary conditions for a steel bar. `T_ref = T0` means the bar is stress-free at its initial uniform temperature (100°C). `dT = 400` is the temperature rise from left to right. All values use SI units.

---

## Step 3: Scaling Infrastructure

```python
T_scale = max(abs(dT), 1.0)
strain_scale = max(abs(alpha * (T0 - T_ref)), abs(alpha * (TL - T_ref)), 1e-12)
u_scale = strain_scale * L
curvature_scale = strain_scale / L
```

This is one of the key improvements over a naive implementation. Instead of using raw physical values (which span many orders of magnitude), we define **characteristic scales**:

| Scale | Value | What it represents |
|-------|-------|-------------------|
| `T_scale` | 400 | Typical temperature variation across the bar |
| `strain_scale` | 4.8×10⁻³ | Maximum thermal strain `α(TL − T_ref)` |
| `u_scale` | 4.8×10⁻³ m | Characteristic displacement magnitude |
| `curvature_scale` | 4.8×10⁻³ /m | Characteristic second derivative of `u` |

These will be used to **normalize the PDE residuals** so that all loss terms live on a similar numerical scale — no need for crazy `10¹²` penalty weights.

---

## Step 4: Analytical Solutions

```python
def exact_T(x):
    return T0 + dT * x / L

def exact_u(x):
    return alpha * (T0 - T_ref) * x + alpha * dT * x**2 / (2 * L)
```

Because the governing equations have closed-form solutions for this specific problem, we define them here for validation.

**Temperature** is linear: steady-state conduction with no heat source means `T'' = 0`, so `T(x)` is just a straight line connecting the boundary values.

**Displacement** is quadratic. The derivation:

1. From `dσ/dx = 0` and `σ(L) = 0` → stress is zero **everywhere** in the bar.
2. Zero stress means: `σ = E(u' − α(T − T_ref)) = 0` → `u'(x) = α(T(x) − T_ref)`
3. Substituting the linear T: `u'(x) = α((T0 − T_ref) + dT·x/L)`
4. Integrating with `u(0) = 0`: `u(x) = α(T0 − T_ref)·x + α·dT·x²/(2L)`

Since `T_ref = T0`, the first term vanishes and we get the simpler parabola: `u(x) = α·dT·x²/(2L)`.

At `x = L`: `u(1) = 12×10⁻⁶ × 400 × 1² / 2 = 2400 μm`.

---

## Step 5: Neural Network Architecture

```python
class CoupledPINN(nn.Module):
    def __init__(self, n_hidden=4, n_neurons=64):
        super().__init__()
        layers = [nn.Linear(1, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        self.trunk = nn.Sequential(*layers)
        self.head_T = nn.Linear(n_neurons, 1)
        self.head_u = nn.Linear(n_neurons, 1)

    def forward(self, x):
        s = x / L                        # normalize input to [0, 1]
        h = self.trunk(s)
        raw_T = self.head_T(h)
        raw_u = self.head_u(h)

        # Hard boundary constraints
        T = T0 + dT * (s + s * (1 - s) * raw_T)
        u = u_scale * s * raw_u

        return T, u
```

The network takes a single input (the spatial coordinate `x`) and produces two outputs: temperature `T(x)` and displacement `u(x)`.

### Architecture choices:

- **Shared trunk** (4 hidden layers × 64 neurons): Learns spatial features common to both fields. Deeper than before (was 40 neurons) because the hard constraints add a small overhead.
- **Separate output heads**: Each physics field gets its own final linear layer.
- **`tanh` activation**: Chosen because it's infinitely differentiable — critical since we compute second derivatives through autograd.
- **Input normalization** (`s = x/L`): Maps the input to `[0, 1]` regardless of bar length.

### Hard Boundary Constraints — The Key Innovation:

Instead of penalizing boundary violations in the loss function (soft enforcement), we **build the boundary conditions directly into the network output**:

**Temperature:**
```
T = T0 + dT * (s + s*(1-s)*raw_T)
```
- At `s = 0`: `T = T0 + dT * 0 = T0 = 100` ✓ (regardless of `raw_T`)
- At `s = 1`: `T = T0 + dT * (1 + 0) = T0 + dT = 500` ✓ (regardless of `raw_T`)
- The `s*(1-s)` factor is a **bubble function** — it's zero at both endpoints and nonzero inside, so the network can only modify the interior of the solution.
- For the exact solution (`T'' = 0`), the network just needs to output `raw_T ≈ 0` everywhere. This is trivial to learn.

**Displacement:**
```
u = u_scale * s * raw_u
```
- At `s = 0`: `u = u_scale * 0 * raw_u = 0` ✓ (fixed end enforced exactly)
- The `u_scale` factor brings the output into the correct physical magnitude.
- For the exact solution, the network needs `raw_u(1) ≈ 0.5` — well within the tanh output range.

**Why this matters:** The optimizer never has to "fight" to satisfy boundary conditions. They're guaranteed from epoch 1. All training effort goes toward satisfying the PDEs in the interior.

---

## Step 6: Optimizer, Scheduler, and Best-State Tracking

```python
model = CoupledPINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

best_loss = float('inf')
best_state = None
```

- **Adam optimizer** at `lr=1e-3`: The standard choice for PINNs — adaptive learning rates handle the multi-scale loss landscape.
- **StepLR scheduler**: Halves the learning rate every 5000 epochs. Aggressive early exploration → fine-tuning later.
- **Best-state tracking**: We save the model weights whenever the loss hits a new minimum. This protects against late-epoch instability — sometimes the loss spikes near the end of training, and you don't want to evaluate on a bad final checkpoint.

---

## Step 7: Collocation Points

```python
n_int = 100
x_int = torch.linspace(0, L, n_int, device=device).reshape(-1, 1)
x_int.requires_grad_(True)
xL_bc = torch.tensor([[L]], device=device, requires_grad=True)
```

100 uniformly spaced points span the domain `[0, L]`. These are the **collocation points** where the PDE residual will be evaluated. `requires_grad_(True)` tells PyTorch to track derivatives of the network output with respect to `x` — this is what enables automatic differentiation of `T` and `u` with respect to spatial coordinates.

`xL_bc` is a single point at the right end for evaluating the stress-free boundary condition (the one soft BC remaining).

**Note**: We don't need boundary points for T(0), T(L), or u(0) because those are **hard-coded** into the network output (Step 5).

---

## Step 8: Training Loop — PDE Residuals

```python
T, u = model(x_int)

# --- Heat equation: k*T'' = 0 ---
T_x = torch.autograd.grad(T, x_int, torch.ones_like(T), create_graph=True)[0]
T_xx = torch.autograd.grad(T_x, x_int, torch.ones_like(T_x), create_graph=True)[0]
r_heat = (k * T_xx) / (k * T_scale / L**2)
loss_heat = torch.mean(r_heat**2)

# --- Elasticity: E*u'' - E*alpha*T' = 0 ---
u_x = torch.autograd.grad(u, x_int, torch.ones_like(u), create_graph=True)[0]
u_xx = torch.autograd.grad(u_x, x_int, torch.ones_like(u_x), create_graph=True)[0]
r_mech = E_mod * (u_xx - alpha * T_x) / (E_mod * curvature_scale)
loss_mech = torch.mean(r_mech**2)
```

This is the physics engine of the PINN. At each collocation point:

1. The network predicts `T` and `u` (with hard BCs already built in).
2. `torch.autograd.grad` computes exact derivatives via the chain rule.
3. These derivatives are substituted into the governing PDEs to compute **residuals** — how much the current prediction violates the physics.
4. The mean squared residual becomes the physics loss.

### Normalization of residuals:

The raw heat residual `k·T''` has units `[W/m³]` and the raw elasticity residual `E·(u'' − α·T')` has units `[Pa/m]`. These live on wildly different scales. We normalize each by its characteristic magnitude:

| Residual | Normalization factor | Result |
|----------|---------------------|--------|
| Heat: `k·T''` | `k·T_scale/L² = 50×400/1 = 20000` | Dimensionless, O(1) when satisfied |
| Elasticity: `E·(u''−α·T')` | `E·curvature_scale = 200×10⁹ × 4.8×10⁻³ = 9.6×10⁸` | Dimensionless, O(1) when satisfied |

Both loss terms now contribute equally without needing manually-tuned weights.

The `create_graph=True` flag is critical: it keeps the computational graph alive so that gradients can flow **through** the derivative computation during backpropagation.

### The coupling:
The temperature gradient `T_x` appears in the elasticity residual. This is how the thermal field drives the mechanical response — the network must learn a displacement field whose curvature `u''` matches `α·T'`.

---

## Step 9: Stress-Free Boundary Condition (Only Soft BC)

```python
_, u_bc = model(xL_bc)
u_x_bc = torch.autograd.grad(u_bc, xL_bc, torch.ones_like(u_bc), create_graph=True)[0]
sigma_L = E_mod * (u_x_bc - alpha * (TL - T_ref))
loss_bc = (sigma_L / (E_mod * strain_scale))**2
```

This is the **only remaining soft constraint** — everything else is hard-coded. We enforce the stress-free condition at the right end: `σ(L) = E(u'(L) − α(T(L) − T_ref)) = 0`.

Why can't this be hard-coded like the others?
- Temperature BCs are Dirichlet (value at a point) — easy to build into a function form.
- `u(0) = 0` is Dirichlet — just multiply by `s`.
- But `σ(L) = 0` is a **Neumann-type** condition on the derivative. Hard-coding derivative constraints is possible but much more complex, so we keep this one as a penalty.

The normalization `E_mod * strain_scale` brings the stress into dimensionless form. The penalty weight is just `100` (in the total loss) — no need for `10¹²` since the term is already properly scaled.

---

## Step 10: Loss Assembly and Best-State Update

```python
loss = loss_heat + loss_mech + 100 * loss_bc

optimizer.zero_grad()
loss.backward()
optimizer.step()
scheduler.step()

if loss.item() < best_loss:
    best_loss = loss.item()
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
```

The total loss has three terms:
- `loss_heat`: Heat equation residual (normalized)
- `loss_mech`: Elasticity equation residual (normalized)
- `100 * loss_bc`: Stress-free BC penalty (moderate weight since it's already normalized)

The weight of `100` on the BC term ensures the boundary condition is prioritized during training, but since the residuals are already normalized, we don't need absurd multipliers.

**Best-state tracking**: After each optimizer step, we compare the current loss to our best-ever loss. If it's lower, we save a deep copy of the model weights. This means our final evaluation uses the best model from **any** point during training, not just the final epoch (which might have spiked due to learning rate schedule or numerical noise).

---

## Step 11: Post-Training — Load Best Model and Evaluate

```python
model.load_state_dict(best_state)
model.eval()

x_test = torch.linspace(0, L, 200, device=device).reshape(-1, 1)
x_test.requires_grad_(True)

with torch.no_grad():
    T_pred, u_pred = model(x_test)
```

After training completes:
1. **Load the best checkpoint** — not the final-epoch weights.
2. Switch to **eval mode** (disables dropout/batchnorm if present; good practice even if not used here).
3. Evaluate on a **denser grid** (200 points vs 100 training points) to test generalization.
4. `torch.no_grad()` disables gradient tracking for inference — saves memory and speeds up evaluation.

The predictions are then compared against the analytical solutions to quantify accuracy.

---

## Step 12: Visualization

The three-panel plot shows:

1. **Temperature field** — PINN vs exact (should overlap perfectly for this linear problem)
2. **Displacement field** — PINN vs exact (quadratic curve, max 2400 μm at free end)
3. **Training convergence** — loss vs epoch on a log scale

The error metrics printed:
- Maximum absolute error for T (expect < 1°C)
- Maximum absolute error for u (expect < 50 μm)
- These validate that the PINN successfully learned both physics simultaneously.

---

## Summary: Training Flow

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         Each Epoch                                   │
│                                                                     │
│   x_int ──→ model(x_int) ──→ T, u  (BCs already satisfied)        │
│                                 │                                   │
│                    ┌────────────┼─────────────┐                     │
│                    ▼            ▼             ▼                      │
│              autograd(T)   autograd(u)   model(xL_bc)              │
│              T', T''       u', u''       u'(L)                      │
│                    │            │             │                      │
│                    ▼            ▼             ▼                      │
│              k·T''/scale   (u''-α·T')/scale  σ(L)/scale            │
│              (normalized)   (normalized)     (normalized)           │
│                    │            │             │                      │
│                    └──────┬─────┘─────────────┘                     │
│                           ▼                                         │
│                      Total Loss                                     │
│                           │                                         │
│                           ▼                                         │
│                    loss.backward()                                  │
│                    optimizer.step()                                  │
│                    ──── save if best ────                           │
│                                                                     │
│   Repeat × 15,000 epochs                                           │
│   Final model = best checkpoint                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Differences from a Naive PINN Implementation

| Aspect | Naive Approach | This Implementation |
|--------|---------------|-------------------|
| Boundary conditions | Soft penalties with large weights (10¹²) | Hard-coded into network output |
| Residual scaling | Raw physics values (10⁹ Pa) | Normalized to O(1) |
| Loss weights | Manual tuning, problem-specific | Only one weight (100 for σ BC) |
| Model selection | Use final epoch | Best checkpoint during training |
| Network input | Raw `x` | Normalized `s = x/L` |

These improvements make the training more stable, faster to converge, and less sensitive to hyperparameter choices.
