# CODE WALKTHROUGH: Heat Conduction with Internal Generation PINN

## Project Overview

This PINN solves the 1D steady-state heat conduction equation with uniform internal
heat generation using SOFT boundary constraints. The key pedagogical goal is to
demonstrate how the boundary penalty weight affects solution accuracy -- running the
same problem twice with weak (weight=1) and strong (weight=100) enforcement.

**Governing Equation:**

```
k * T''(x) + Q = 0,   x in [0, L]
T(0) = 100,  T(L) = 200
```

**Physical Meaning:** A rod of length L with thermal conductivity k generates heat
internally at rate Q (e.g., electrical resistance heating, nuclear fuel rod). The
temperature distribution balances diffusion against generation.

---

## Section 1: Imports and Device Setup

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**What each import does:**

| Import | Role |
|--------|------|
| torch | Tensor operations, autograd for derivatives |
| torch.nn | Neural network layers and modules |
| numpy | Post-processing arrays, analytical solution |
| matplotlib | Three-panel comparison plot |

**Seed = 42:** Both training runs use the same seed to ensure fair comparison.
Any difference in results comes purely from the boundary weight, not random
initialization luck.

**Device selection:** CUDA acceleration if available, otherwise CPU. For this
small network (3x30), CPU is typically sufficient.

---

## Section 2: Physical Parameters

```python
k = 50.0       # Thermal conductivity [W/(m*K)]
Q = 1000.0     # Internal heat generation rate [W/m^3]
L = 1.0        # Domain length [m]
T0 = 100.0     # Left boundary temperature [K or C]
TL = 200.0     # Right boundary temperature [K or C]
```

**Physical interpretation:**

- k=50 is typical of a carbon steel rod
- Q=1000 represents moderate volumetric heating (electrical current, chemical reaction)
- L=1.0 means a 1-meter rod
- Dirichlet BCs: fixed temperatures at both ends (thermal reservoirs)

**Why these values matter for training:** The ratio Q*L^2/k = 1000*1/50 = 20 degrees
of temperature rise due to generation. Combined with the 100-degree difference between
boundaries, the solution has both a linear trend and a parabolic bulge.

---

## Section 3: Scaling and Nondimensionalization

```python
T_scale = max(abs(TL - T0), 1.0)   # = max(100, 1) = 100
theta_left = (T0 - T0) / T_scale   # = 0
theta_right = (TL - T0) / T_scale  # = 1
source = Q * L**2 / (k * T_scale)  # = 1000 * 1 / (50 * 100) = 0.2
```

**Derivation of the nondimensional form:**

Starting from the physical equation:

```
k * T''(x) + Q = 0
```

Define nondimensional variables:

```
s = x / L              (spatial coordinate, s in [0, 1])
theta = (T - T0) / T_scale   (temperature, near O(1))
```

Then T = T0 + T_scale * theta, and:

```
dT/dx = (T_scale / L) * d(theta)/ds
d^2T/dx^2 = (T_scale / L^2) * d^2(theta)/ds^2
```

Substituting into the PDE:

```
k * (T_scale / L^2) * theta''(s) + Q = 0
```

Dividing by k * T_scale / L^2:

```
theta''(s) + Q * L^2 / (k * T_scale) = 0
theta''(s) + source = 0
```

Where source = Q * L^2 / (k * T_scale) = 0.2.

**Nondimensional boundary conditions:**

```
theta(0) = (T0 - T0) / T_scale = 0
theta(1) = (TL - T0) / T_scale = 1
```

**Why this scaling helps:**

| Quantity | Physical | Nondimensional |
|----------|----------|----------------|
| Domain | [0, 1.0] m | [0, 1] |
| Temperature | [100, 210] K | [0, 1.1] |
| Source term | 1000 W/m^3 | 0.2 |
| Left BC | 100 | 0 |
| Right BC | 200 | 1 |

Everything is O(1), preventing gradient imbalance during training.

---

## Section 4: Analytical Solution

```python
def exact_temperature(x):
    return -(Q / (2 * k)) * x**2 + ((TL - T0) / L + Q * L / (2 * k)) * x + T0
```

**Full derivation:**

Starting from k * T''(x) + Q = 0:

```
T''(x) = -Q/k
```

Integrate once:

```
T'(x) = -(Q/k) * x + C1
```

Integrate again:

```
T(x) = -(Q/(2k)) * x^2 + C1 * x + C2
```

Apply BC at x=0:

```
T(0) = C2 = T0 = 100
```

Apply BC at x=L:

```
T(L) = -(Q/(2k)) * L^2 + C1 * L + T0 = TL
C1 = (TL - T0) / L + Q * L / (2k)
C1 = (200 - 100) / 1 + 1000 * 1 / (2 * 50)
C1 = 100 + 10 = 110
```

**Final analytical solution:**

```
T(x) = -10 * x^2 + 110 * x + 100
```

**Key features:**

- Parabola opening downward (generation creates a temperature bulge)
- Maximum at x* = C1 * k / Q = 110 * 50 / 1000 = 5.5 (outside domain!)
- Within [0,1]: monotonically increasing from 100 to 200
- Temperature at midpoint: T(0.5) = -10*0.25 + 110*0.5 + 100 = 152.5

---

## Section 5: Derivative Helper

```python
def derivative(y, x, order=1):
    dy = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                             create_graph=True)[0]
    if order == 1:
        return dy
    return derivative(dy, x, order - 1)
```

**How it works for this problem:**

The PDE needs theta''(s), so we call derivative(theta, s, order=2):

1. First call: computes d(theta)/ds via autograd
2. Recursive call: computes d^2(theta)/ds^2 from the first derivative

**create_graph=True** is essential -- it keeps the computation graph alive so that:
- Higher-order derivatives can be computed (second derivative needs graph of first)
- The final loss can backpropagate through the derivative operations

---

## Section 6: Network Architecture

```python
class HeatNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, 1)
        )

    def forward(self, s):
        return self.net(s)
```

**Architecture: 3 hidden layers, 30 neurons each, tanh activation.**

**Critical design choice -- NO hard boundary constraint:**

```python
def forward(self, s):
    return self.net(s)    # Raw network output, nothing else
```

Compare with a hard-constraint approach that would look like:

```python
# NOT used here -- this is what hard enforcement would look like:
def forward(self, s):
    raw = self.net(s)
    return theta_left * (1 - s) + theta_right * s + s * (1 - s) * raw
```

**WHY soft constraints were chosen for this project:**

| Reason | Explanation |
|--------|-------------|
| Pedagogical | Demonstrates weight sensitivity -- the central lesson |
| Comparison study | Same architecture, different weights, different results |
| Realistic scenario | Many real problems lack easy hard-constraint forms |
| Failure mode demo | Shows what happens when weight is too low |

The whole point of this project is to show that soft enforcement is NOT automatic --
the user must tune the penalty weight. This teaches a fundamental PINN lesson.

**Parameter count:**

```
Layer 1: 1*30 + 30 = 60
Layer 2: 30*30 + 30 = 930
Layer 3: 30*30 + 30 = 930
Layer 4: 30*1 + 1 = 31
Total: 1951 parameters
```

---

## Section 7: train_heat_pinn() Function Design

```python
def train_heat_pinn(boundary_weight, epochs=6000):
    torch.manual_seed(42)
    model = HeatNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)
    ...
```

**Why a function-based design:**

This project runs the SAME training procedure TWICE with different boundary weights.
Wrapping everything in a function ensures:

1. **Fresh model each time** -- torch.manual_seed(42) before HeatNet() gives identical
   initial weights for both runs
2. **Fair comparison** -- same optimizer, same schedule, same collocation points
3. **Clean encapsulation** -- each run returns a results dictionary
4. **Reproducibility** -- calling train_heat_pinn(w) always produces the same result

**Optimizer and scheduler:**

```
Adam, lr=1e-3          -- standard PINN learning rate
StepLR:
  step_size=3000       -- halve LR at epoch 3000
  gamma=0.5            -- new lr = 5e-4 for epochs 3001-6000
```

The LR schedule provides aggressive initial exploration followed by fine-tuning.
At epoch 3000 (halfway), the learning rate drops to allow convergence to a
more precise minimum.

---

## Section 8: Collocation Points and Boundary Setup

```python
    s_colloc = torch.linspace(0, 1, 80, device=device).reshape(-1, 1).requires_grad_(True)

    s_bc = torch.tensor([[0.0], [1.0]], device=device)
    theta_bc = torch.tensor([[0.0], [1.0]], device=device)  # theta_left=0, theta_right=1
```

**Collocation points (80 interior points):**

- Uniformly spaced in [0, 1]
- requires_grad_(True) enables autograd to compute theta' and theta''
- 80 points for a 1D problem is generous -- ensures smooth derivative estimates

**Boundary points:**

- s_bc contains the two boundary locations: s=0 and s=1
- theta_bc contains the target nondimensional temperatures: 0 and 1
- These are SEPARATE from collocation points -- they form their own loss term

**Why separate BC points matter for soft enforcement:**

With hard constraints, BCs are satisfied by construction. With soft constraints,
the network must learn to match BCs through gradient descent. Having explicit
BC tensors creates a clear, direct penalty signal.

---

## Section 9: Training Loop

```python
    losses = []
    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        optimizer.zero_grad()

        theta = model(s_colloc)
        theta_s = derivative(theta, s_colloc, order=1)
        theta_ss = derivative(theta, s_colloc, order=2)

        loss_pde = torch.mean((theta_ss + source) ** 2)
        loss_bc = torch.mean((model(s_bc) - theta_bc) ** 2)
        loss = loss_pde + boundary_weight * loss_bc

        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = model.state_dict().copy()

    model.load_state_dict(best_state)
```

**Training Flow Diagram:**

```
+------------------------------------------------------------------+
|                    TRAINING LOOP (6000 epochs)                    |
+------------------------------------------------------------------+
|                                                                    |
|  s_colloc [80x1]                    s_bc [2x1]                    |
|       |                                  |                         |
|       v                                  v                         |
|  +----------+                      +----------+                    |
|  | HeatNet  |                      | HeatNet  |                    |
|  +----------+                      +----------+                    |
|       |                                  |                         |
|       v                                  v                         |
|   theta [80x1]                    theta_pred [2x1]                 |
|       |                                  |                         |
|       v                                  v                         |
|  autograd (order=2)              compare with theta_bc             |
|       |                                  |                         |
|       v                                  v                         |
|   theta_ss [80x1]               loss_bc = MSE(pred - target)      |
|       |                                  |                         |
|       v                                  |                         |
|  loss_pde = mean(                        |                         |
|    (theta_ss + 0.2)^2                    |                         |
|  )                                       |                         |
|       |                                  |                         |
|       +----------------+  +--------------+                         |
|                        |  |                                        |
|                        v  v                                        |
|          loss = loss_pde + weight * loss_bc                        |
|                           |                                        |
|                           v                                        |
|                    loss.backward()                                  |
|                           |                                        |
|                           v                                        |
|                    optimizer.step()                                 |
|                    scheduler.step()                                 |
|                           |                                        |
|                           v                                        |
|                  Track best_state if                                |
|                  loss < best_loss                                   |
|                                                                    |
+------------------------------------------------------------------+
```

**Loss components explained:**

1. **PDE Loss (loss_pde):**
   ```
   loss_pde = mean((theta_ss + source)^2) = mean((theta_ss + 0.2)^2)
   ```
   Enforces the nondimensional PDE: theta''(s) + 0.2 = 0 at all 80 collocation points.

2. **Boundary Loss (loss_bc):**
   ```
   loss_bc = mean((model(s_bc) - theta_bc)^2)
   ```
   Enforces theta(0)=0 and theta(1)=1. This is a 2-point MSE.

3. **Total Loss (weighted sum):**
   ```
   loss = loss_pde + boundary_weight * loss_bc
   ```
   The boundary_weight controls how strongly the optimizer prioritizes BC satisfaction
   over PDE satisfaction.

**What boundary_weight means physically:**

| Weight | Interpretation | Expected Behavior |
|--------|---------------|-------------------|
| 1 | BCs and PDE equally important | PDE may dominate; BCs approximately satisfied |
| 10 | BCs 10x more important | Better BC satisfaction, still some error |
| 100 | BCs 100x more important | BCs nearly exact; PDE well-satisfied too |
| 1000+ | BCs overwhelmingly dominant | Risk: optimizer ignores PDE to nail BCs |

**Why weight=1 can fail:**

The PDE loss is averaged over 80 points. The BC loss is averaged over 2 points.
Even without the weight, the PDE loss has 40x more "votes" in the gradient.
A weight of 1 means BC satisfaction is an afterthought -- the optimizer finds
it easier to reduce the 80-point PDE residual than to pin 2 boundary values.

**Best-state tracking:**

```python
if loss.item() < best_loss:
    best_loss = loss.item()
    best_state = model.state_dict().copy()
```

Neural network training is noisy. The loss may fluctuate, especially after the
LR drop at epoch 3000. Tracking the best state ensures we keep the globally
best model seen during training, not just the final (potentially worse) state.

---

## Section 10: Post-Training Evaluation

```python
    s_eval = torch.linspace(0, 1, 200, device=device).reshape(-1, 1).requires_grad_(True)
    theta_eval = model(s_eval)
    theta_ss_eval = derivative(theta_eval, s_eval, order=2)

    # Convert to physical units
    x_phys = s_eval.detach().cpu().numpy().flatten() * L
    temperature = T0 + T_scale * theta_eval.detach().cpu().numpy().flatten()
    exact = exact_temperature(x_phys)

    # Physical PDE residual
    residual = (k * T_scale / L**2) * theta_ss_eval.detach().cpu().numpy().flatten() + Q
```

**Evaluation on 200 points** (denser than 80 training points) to get smooth curves.

**Converting nondimensional back to physical:**

```
x = s * L                          (spatial)
T = T0 + T_scale * theta           (temperature)
T = 100 + 100 * theta
```

**Physical residual computation:**

Starting from T = T0 + T_scale * theta and x = s * L:

```
d^2T/dx^2 = (T_scale / L^2) * d^2(theta)/ds^2
```

So the physical PDE residual is:

```
R(x) = k * d^2T/dx^2 + Q
     = k * (T_scale / L^2) * theta_ss + Q
     = 50 * (100 / 1) * theta_ss + 1000
     = 5000 * theta_ss + 1000
```

If the PDE is perfectly satisfied, R(x) = 0 everywhere. Nonzero residual
shows where the network solution deviates from the true physics.

**Metrics computed:**

```python
    max_error = np.max(np.abs(temperature - exact))
    mean_error = np.mean(np.abs(temperature - exact))
    max_residual = np.max(np.abs(residual))
```

These quantify solution quality in physical units (degrees and W/m^3).

---

## Section 11: Comparison Run

```python
# Main execution
print("Training with WEAK boundary enforcement (weight=1)...")
weak = train_heat_pinn(boundary_weight=1)

print("Training with STRONG boundary enforcement (weight=100)...")
strong = train_heat_pinn(boundary_weight=100)

# Print comparison
print(f"\n{'Metric':<25} {'Weight=1':<15} {'Weight=100':<15}")
print("-" * 55)
print(f"{'Max Error [K]':<25} {weak['metrics']['max_error']:<15.4f} {strong['metrics']['max_error']:<15.4f}")
print(f"{'Mean Error [K]':<25} {weak['metrics']['mean_error']:<15.4f} {strong['metrics']['mean_error']:<15.4f}")
print(f"{'Max Residual [W/m^3]':<25} {weak['metrics']['max_residual']:<15.4f} {strong['metrics']['max_residual']:<15.4f}")
```

**What we expect to see:**

| Metric | Weight=1 (Weak) | Weight=100 (Strong) |
|--------|-----------------|---------------------|
| Max Error | Several degrees | Sub-degree |
| Mean Error | ~1-5 K | ~0.01-0.1 K |
| Max Residual | Moderate | Small |
| BC Error | Visible offset | Negligible |

**Why weight=1 underperforms:**

With weight=1, the optimizer treats boundary satisfaction as just 2 more data
points among 80. The gradient from loss_pde (80 points) overwhelms the gradient
from loss_bc (2 points scaled by 1). The network learns a good PDE shape but
"floats" -- it may satisfy theta'' + 0.2 = 0 approximately while being offset
from the correct boundary values.

**Why weight=100 works well:**

With weight=100, the BC gradient is amplified 100x. Now the optimizer MUST
satisfy the boundaries before it can reduce total loss. Once boundaries are
pinned, the PDE loss guides the interior solution to the correct parabola.
This mimics what hard constraints achieve automatically.

**The sweet spot:**

- Too low (weight=1): BCs not enforced, solution drifts
- Just right (weight=100): BCs enforced, PDE also satisfied
- Too high (weight=10000+): Optimizer spends all effort on BCs, PDE interior suffers

---

## Section 12: Visualization

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Panel 1: Temperature comparison
axes[0].plot(weak['x'], weak['exact'], 'k-', linewidth=2, label='Exact')
axes[0].plot(weak['x'], weak['temperature'], 'b--', linewidth=1.5, label='Weight=1')
axes[0].plot(strong['x'], strong['temperature'], 'r--', linewidth=1.5, label='Weight=100')
axes[0].set_xlabel('x [m]')
axes[0].set_ylabel('Temperature [K]')
axes[0].set_title('Temperature Distribution')
axes[0].legend()

# Panel 2: PDE residual
axes[1].plot(weak['x'], weak['residual'], 'b-', label='Weight=1')
axes[1].plot(strong['x'], strong['residual'], 'r-', label='Weight=100')
axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
axes[1].set_xlabel('x [m]')
axes[1].set_ylabel('Residual [W/m^3]')
axes[1].set_title('PDE Residual: k*T\'\' + Q')
axes[1].legend()

# Panel 3: Loss convergence
axes[2].semilogy(weak['losses'], 'b-', alpha=0.7, label='Weight=1')
axes[2].semilogy(strong['losses'], 'r-', alpha=0.7, label='Weight=100')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Total Loss')
axes[2].set_title('Training Convergence')
axes[2].legend()

plt.tight_layout()
plt.savefig('heat_pinn_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Three-panel layout:**

```
+-------------------+-------------------+-------------------+
|                   |                   |                   |
|   Temperature     |   PDE Residual    |   Convergence     |
|   Distribution    |                   |   (log scale)     |
|                   |                   |                   |
|  Black = Exact    |  Blue = Weight=1  |  Blue = Weight=1  |
|  Blue = Weight=1  |  Red = Weight=100 |  Red = Weight=100 |
|  Red = Weight=100 |  Dashed = zero    |                   |
|                   |                   |                   |
+-------------------+-------------------+-------------------+
```

**What each panel reveals:**

1. **Temperature Distribution:** Shows how weight=1 may have an offset or shape
   error compared to the exact parabola, while weight=100 overlaps the exact curve.

2. **PDE Residual:** Shows where in the domain each model violates k*T''+Q=0.
   Ideally zero everywhere. Weight=100 should have smaller, more uniform residual.

3. **Training Convergence:** Shows loss history on log scale. Weight=100 starts
   higher (because BC loss is amplified) but converges to a solution that satisfies
   BOTH the PDE and boundaries. The LR drop at epoch 3000 is visible as a kink.

---

## Comparison Table: All Four PINN Projects

| Feature | Beam Deflection | Heat Conduction | Fluid Flow | Vibration |
|---------|----------------|-----------------|------------|-----------|
| **PDE** | EI*w''''=q | k*T''+Q=0 | mu*u''+dp/dx=0 | u_tt=c^2*u_xx |
| **Order** | 4th | 2nd | 2nd | 2nd (space+time) |
| **Domain** | [0, L] | [0, L] | [0, H] | [0,L] x [0,T] |
| **BC Type** | Hard constraint | Soft constraint | Hard constraint | Hard constraint |
| **Why** | 4 BCs complex | Demonstrate weights | Clean comparison | Natural lifting |
| **Key Lesson** | High-order PINNs | Weight sensitivity | Viscous profiles | Wave propagation |
| **Network** | 4x40, tanh | 3x30, tanh | 4x30, tanh | 4x50, tanh |
| **Training** | 8000 epochs | 6000 epochs x2 | 10000 epochs | 15000 epochs |
| **Outputs** | Deflection w(x) | Temperature T(x) | Velocity u(y) | Displacement u(x,t) |
| **Scaling** | w_scale, q_bar | T_scale, source | u_scale, f_hat | u_scale, c_bar |
| **Comparison** | PINN vs exact | Weak vs strong BC | Laminar vs turbulent | PINN vs exact |
| **Unique** | 4th derivative | Runs twice | Two Re numbers | 2D input (x,t) |

---

## Key Design Decisions Explained

### Why Soft Boundary Constraints Here?

This is a deliberate pedagogical choice. The other three projects use hard constraints,
which automatically satisfy BCs. This project answers the question: "What if we
CANNOT easily construct a hard constraint?"

Real-world scenarios where soft constraints are necessary:

- Complex geometries where lifting functions are hard to define
- Time-dependent BCs that change during simulation
- Integral constraints (e.g., fixed total flux, not pointwise)
- Multi-physics coupling where BCs come from another solver
- Neumann/Robin BCs mixed with Dirichlet

By using soft constraints on a problem where we COULD use hard ones, we can
compare the result against the known-good analytical solution and quantify
exactly how much accuracy is lost (or recovered with proper weighting).

### Why Two Runs Instead of One?

A single training run with one weight tells you nothing about sensitivity.
Running twice with weight=1 and weight=100 demonstrates:

1. The problem is NOT just "add a BC loss and it works"
2. The weight is a hyperparameter that requires tuning
3. There is a trade-off: too low = bad BCs, too high = slow convergence
4. The "right" weight depends on the relative magnitudes of PDE and BC losses

### Why 6000 Epochs (Not More)?

- 6000 is enough for both runs to converge (or reveal convergence failure)
- The StepLR at 3000 provides a natural "coarse then fine" training schedule
- Running longer with weight=1 would NOT fix the fundamental under-weighting issue
- Running longer with weight=100 shows diminishing returns after convergence

### Why 3 Hidden Layers x 30 Neurons?

For a smooth parabolic solution, this is more than sufficient:

- The exact solution is a quadratic polynomial
- A single-layer network with tanh could approximate this
- 3 layers provide robustness and fast convergence
- 30 neurons per layer (~1951 params) is modest and trains quickly

The network is intentionally NOT the bottleneck -- any failure to match the
exact solution comes from the soft constraint formulation, not network capacity.

---

## Understanding the Loss Landscape

**With weight=1:**

```
Total Loss = loss_pde + 1 * loss_bc

The optimizer sees:
  - 80 PDE residual terms contributing to gradients
  - 2 BC residual terms contributing to gradients (unscaled)
  - Result: optimizer naturally prioritizes reducing PDE residual
  - BCs get "whatever is left over"
```

**With weight=100:**

```
Total Loss = loss_pde + 100 * loss_bc

The optimizer sees:
  - 80 PDE residual terms contributing to gradients
  - 2 BC residual terms, each amplified 100x in the gradient
  - Result: BC gradient is now comparable to total PDE gradient
  - Optimizer must satisfy BCs to make progress
```

**Gradient magnitude comparison:**

```
grad(loss_pde) ~ sum of 80 terms ~ O(80 * pde_error^2)
grad(loss_bc, weight=1) ~ sum of 2 terms ~ O(2 * bc_error^2)
grad(loss_bc, weight=100) ~ sum of 2*100 terms ~ O(200 * bc_error^2)
```

With weight=100, the BC gradient is now 200/80 = 2.5x the PDE gradient per unit
error, ensuring boundaries are prioritized.

---

## Data Flow Summary

```
INPUT                    PROCESSING                     OUTPUT
-----                    ----------                     ------

boundary_weight    --->  train_heat_pinn()  --->  results dict:
(1 or 100)               |                          - model (trained)
                          |                          - x (physical coords)
                          |-- Create model            - temperature (PINN)
                          |-- Train 6000 epochs       - exact (analytical)
                          |-- Evaluate on 200 pts     - residual (PDE error)
                          |-- Convert to physical     - losses (history)
                          |-- Compute metrics         - metrics (errors)
                          |
                          v
                    Return results dict

Main script:
  weak = train_heat_pinn(1)      # First run
  strong = train_heat_pinn(100)  # Second run
  compare_and_plot(weak, strong) # Side-by-side analysis
```

---

## Common Issues and Debugging

**Q: Weight=1 gives large BC error but small PDE residual. Why?**

A: Expected behavior. The optimizer found it efficient to reduce the 80-point
PDE loss while ignoring the 2-point BC loss. The network output "floats" --
it has the right shape (parabola) but wrong vertical position.

**Q: Weight=100 loss starts higher than weight=1. Is that wrong?**

A: No. The initial BC error gets multiplied by 100, so the initial total loss
is ~100x larger. But this large initial loss creates strong gradients toward
BC satisfaction, leading to a better final solution.

**Q: Could we use adaptive weighting instead of fixed weights?**

A: Yes! Methods like learning-rate annealing (e.g., Wang et al. 2021) or
self-adaptive weights (McClenny & Brundage 2020) automatically balance loss
components. This project uses fixed weights for clarity and reproducibility.

**Q: Why not just increase collocation points near boundaries?**

A: That is another valid strategy (boundary-concentrated sampling). However,
it changes the problem setup rather than addressing the fundamental question
of loss weighting. This project isolates the weight effect specifically.

---

## Execution Checklist

1. Run the script -- it trains twice automatically
2. Compare printed metrics: weight=100 should win on all counts
3. Check the 3-panel plot:
   - Panel 1: weight=100 curve should overlap exact (black)
   - Panel 2: weight=100 residual should be smaller
   - Panel 3: Both should converge, but to different final values
4. Note the BC errors specifically -- weight=1 may show 5-10 degree offset at boundaries

---

## File Structure

```
heat_conduction_pinn/
|-- heat_pinn.py              # Complete script (single file)
|-- heat_pinn_comparison.png  # Output: 3-panel comparison plot
|-- CODE_WALKTHROUGH.md       # This document
```

The entire project is self-contained in one Python file. No external data files,
no configuration, no dependencies beyond PyTorch/NumPy/Matplotlib. Run it and
observe the effect of boundary weight on PINN accuracy.
