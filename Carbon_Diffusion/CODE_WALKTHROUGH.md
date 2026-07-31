# CODE WALKTHROUGH: Carbon Diffusion PINN

## What This Project Does

This project uses a Physics-Informed Neural Network (PINN) to solve the one-dimensional diffusion equation for carbon atoms penetrating into a metal surface. The network learns to predict carbon concentration as a function of both position and time, matching the physics of Fick's second law without ever seeing simulation data.

The key difference from simpler PINN projects: this is a **partial differential equation (PDE)**, not an ordinary differential equation (ODE). The unknown concentration C depends on two independent variables (position x and time t), making the problem fundamentally harder. The network must learn spatial gradients, temporal evolution, and their coupling all at once.

---

## The Physical Problem

Carbon case hardening is an industrial process where carbon atoms diffuse into steel at high temperature. The governing equation is Fick's second law:

```
dC/dt = D * d2C/dx2
```

where:
- C(x,t) is carbon concentration (normalized 0 to 1)
- D = 1e-11 m2/s is the diffusion coefficient
- x is depth into the material (0 = surface, L = 0.002 m)
- t is time (0 to 3600 seconds = 1 hour)

Boundary conditions:
- C(0, t) = 1 for t > 0 (surface held at constant concentration)
- C(L, t) = 0 (far boundary remains clean)
- C(x, 0) = 0 for x > 0 (initially no carbon inside)

The exact solution involves the complementary error function: C(x,t) = erfc(x / (2*sqrt(D*t)))

---

## Step-by-Step Code Breakdown

### Step 1: Imports and Setup

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

Standard PINN stack. The `erfc` import from scipy is used only for generating the analytical solution to compare against. The network never sees this function during training.

Setting the seed to 42 ensures reproducible Sobol sequences and weight initialization.

---

### Step 2: Physical Parameters and Nondimensionalization

```python
D = 1e-11          # Diffusion coefficient [m^2/s]
L = 0.002          # Domain length [m]
t_final = 3600     # Final time [s]
t_star_final = D * t_final / L**2   # = 9e-3
```

**Why nondimensionalize?** The raw numbers span wildly different scales. Position is in millimeters, time in thousands of seconds, and the diffusion coefficient is 1e-11. Neural networks struggle when inputs and outputs differ by many orders of magnitude.

The nondimensionalization scheme:
- x* = x / L, so x* ranges from 0 to 1
- tau = t / t_final, so tau ranges from 0 to 1
- C stays between 0 and 1 (already normalized)

**How the PDE transforms:** Starting from dC/dt = D * d2C/dx2, apply the chain rule:

```
dC/d(tau) * (1/t_final) = D * d2C/d(x*)^2 * (1/L^2)

dC/d(tau) = (D * t_final / L^2) * d2C/d(x*)^2

dC/d(tau) = t_star_final * d2C/d(x*)^2
```

So `t_star_final = 9e-3` is the single dimensionless number controlling this problem. It represents how far diffusion penetrates relative to the domain size during the total time. A small value (much less than 1) means diffusion only reaches a fraction of the domain.

---

### Step 3: Analytical Solution

```python
def analytical_concentration(x, t):
    """Exact solution using complementary error function."""
    if t == 0:
        return np.zeros_like(x)
    return erfc(x / (2 * np.sqrt(D * t)))
```

**What is erfc?** The complementary error function erfc(z) = 1 - erf(z), where erf is the integral of the Gaussian bell curve. It appears naturally in diffusion problems because the fundamental solution to the heat equation is a Gaussian that spreads over time.

The argument x/(2*sqrt(D*t)) is called the "similarity variable." It captures the key physics: concentration profiles at different times collapse onto a single curve when plotted against this variable. The penetration depth grows as sqrt(D*t), not linearly with time.

The t=0 guard prevents division by zero (sqrt(0) in the denominator).

---

### Step 4: Derivative Helper

```python
def derivative(y, x):
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True
    )[0]
```

Same autograd pattern used in all PINN projects. `create_graph=True` is essential because we need second derivatives (C_xx), which means differentiating through the first derivative computation.

For this PDE problem, we call this function three times per training step:
1. C_tau = derivative(C, tau) -- temporal derivative
2. C_x = derivative(C, x) -- first spatial derivative
3. C_xx = derivative(C_x, x) -- second spatial derivative

---

### Step 5: Network Architecture (DiffusionNet)

```python
class DiffusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 40),   # 3 input features
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 1)    # scalar output
        )

    def forward(self, x, tau):
        sqrt_tau = torch.sqrt(tau + 1e-6)
        features = torch.cat([x, tau, sqrt_tau], dim=1)
        raw = self.net(features)
        return (1 - x) * torch.sigmoid(raw)
```

Architecture: [3, 40, 40, 40, 40, 1] with tanh activations.

**Three critical design choices:**

**1. Why sqrt(tau) as an input feature:**
The penetration depth of diffusion scales as sqrt(D*t), which means sqrt(tau) in dimensionless form. By providing this pre-computed feature, we give the network direct access to the natural length scale of the problem. Without it, the network would need to learn this nonlinear relationship from scratch using its hidden layers. The 1e-6 offset prevents gradient issues at tau=0.

**2. Why sigmoid bounds output to [0,1]:**
Carbon concentration is physically bounded. It cannot be negative (no "anti-carbon") and cannot exceed 1 (the surface value is the maximum). The sigmoid function guarantees this constraint regardless of what the internal network produces. This is a hard constraint, not a soft penalty.

**3. Why (1-x) multiplies the output:**
At x=1 (the far boundary in normalized coordinates), concentration must be zero for all time. Multiplying by (1-x) forces the output to exactly zero at x=1, eliminating one boundary condition from the loss function entirely. The network only needs to learn the interior behavior and the surface condition.

---

### Step 6: Optimizer and Scheduler

```python
model = DiffusionNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
```

- Adam at 1e-3: standard starting rate for PINNs
- StepLR halves the learning rate every 5000 epochs
- With 12000 total epochs: rate is 1e-3 for epochs 0-4999, then 5e-4 for 5000-9999, then 2.5e-4 for 10000-11999
- The decay helps fine-tune the solution after the network finds the general shape

---

### Step 7: Sobol Sampling for Interior Points

```python
sobol = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=42)
points = sobol.draw(1000)

# First 500: uniform distribution
x_uniform = points[:500, 0:1]
tau_uniform = points[:500, 1:2]

# Second 500: biased toward origin (squared)
x_biased = points[500:, 0:1] ** 2
tau_biased = points[500:, 1:2] ** 2

x_int = torch.cat([x_uniform, x_biased], dim=0)
tau_int = torch.cat([tau_uniform, tau_biased], dim=0)

# Shift tau away from zero
tau_int = 1e-4 + (1 - 1e-4) * tau_int

x_int.requires_grad_(True)
tau_int.requires_grad_(True)
```

**Why Sobol sequences instead of random sampling?**

Sobol sequences are quasi-random: they fill space more evenly than pseudorandom numbers while avoiding the rigidity of a regular grid. For 1000 points in 2D:
- Random: clumps and gaps due to chance (poor coverage)
- Grid: 31x31 = 961 points locked to specific locations (no adaptivity)
- Sobol: provably low-discrepancy (every sub-region gets fair coverage)

The "scramble=True" adds randomization to prevent correlation artifacts while preserving the space-filling property.

**Why squaring biases toward the origin:**

If u is uniform on [0,1], then u^2 concentrates near 0. The probability density of u^2 is proportional to 1/sqrt(x), so you get many more points near x=0 and tau=0.

This matters because:
- Near x=0 (the surface): the concentration gradient is steepest
- Near tau=0 (early time): the solution changes most rapidly
- The diffusion front is thin and sharp at early times near the surface

Without biasing, most interior points would land in the boring flat region where C is approximately 0. The biased points ensure the network sees enough training signal where the physics is most active.

**Why shift tau by 1e-4:**

At tau=0 exactly, there is a singularity (the initial condition is discontinuous at x=0). Shifting tau slightly avoids numerical issues while still capturing early-time behavior.

---

### Step 8: Boundary and Initial Condition Points

```python
# Surface boundary: x=0, tau varies
tau_bc = torch.linspace(1e-4, 1, 250).reshape(-1, 1).to(device)
x_bc = torch.zeros_like(tau_bc)

# Initial condition: tau=0, x varies
x_ic = torch.linspace(1e-3, 1, 250).reshape(-1, 1).to(device)
tau_ic = torch.zeros_like(x_ic)
```

**Why the corner point (0,0) is excluded from both sets:**

At the corner (x=0, t=0), the boundary condition says C=1 (surface is always at full concentration) but the initial condition says C=0 (material starts clean). This is a mathematical singularity where the solution is discontinuous.

The exact solution erfc(0/(2*sqrt(0))) is undefined (0/0 form). Physically, the concentration jumps instantaneously at the surface when diffusion begins, but no smooth function can represent this.

By starting tau_bc at 1e-4 (not 0) and x_ic at 1e-3 (not 0), we avoid asking the network to fit a discontinuity. The network smoothly interpolates near the corner, which is physically reasonable (in reality, the concentration rises very fast but not instantaneously at the surface).

---

### Step 9: Training Loop

```python
best_loss = float('inf')
best_state = None

for epoch in range(12000):
    optimizer.zero_grad()

    # PDE residual at interior points
    C = model(x_int, tau_int)
    C_tau = derivative(C, tau_int)
    C_x = derivative(C, x_int)
    C_xx = derivative(C_x, x_int)

    residual = C_tau - t_star_final * C_xx
    loss_pde = torch.mean(residual ** 2)

    # Surface BC: C(0, tau) = 1
    C_surface = model(x_bc, tau_bc)
    loss_surface = torch.mean((C_surface - 1.0) ** 2)

    # Initial condition: C(x, 0) = 0
    C_initial = model(x_ic, tau_ic)
    loss_initial = torch.mean(C_initial ** 2)

    # Weighted total loss
    loss = loss_pde + 20 * loss_surface + 20 * loss_initial

    loss.backward()
    optimizer.step()
    scheduler.step()

    # Track best model
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_state = model.state_dict().copy()
```

**The loss weighting (20x for boundaries):**

Without weighting, the PDE residual dominates because it has 1000 points vs 250 for each boundary. But boundary accuracy is more important for the overall solution quality. The factor of 20 ensures the network prioritizes matching the physical constraints.

Think of it as: "I would rather have a slightly imperfect PDE residual everywhere than have the wrong value at the surface."

**Why best-state tracking matters:**

The learning rate decay can occasionally cause the loss to jump when the scheduler steps. By saving the best model seen during all 12000 epochs, we guarantee the final evaluation uses the most accurate weights regardless of late-training instabilities.

---

### Step 10: Post-Training Evaluation

```python
model.load_state_dict(best_state)
model.eval()

# Evaluate at three time snapshots
times = [900, 1800, 3600]  # seconds
x_plot = np.linspace(0, L, 200)
x_norm = torch.tensor(x_plot / L, dtype=torch.float32).reshape(-1, 1).to(device)

for t in times:
    tau_val = t / t_final
    tau_plot = torch.full_like(x_norm, tau_val)

    with torch.no_grad():
        C_pred = model(x_norm, tau_plot).cpu().numpy()

    C_exact = analytical_concentration(x_plot, t)
    # Compare predictions vs exact
```

The evaluation converts back to physical units (meters, seconds) for plotting. Three snapshots at 15 minutes, 30 minutes, and 60 minutes show how the diffusion front penetrates deeper over time.

---

### Step 11: Visualization

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: concentration profiles
ax1.plot(x_plot * 1000, C_exact, 'k-', label='Analytical')
ax1.plot(x_plot * 1000, C_pred, 'r--', label='PINN')
ax1.set_xlabel('Depth [mm]')
ax1.set_ylabel('Concentration')

# Right panel: training convergence
ax2.semilogy(losses)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
```

Two-panel figure: concentration profiles (left) verify physical accuracy, while the convergence plot (right) shows training health. The loss should drop 3-4 orders of magnitude over 12000 epochs with visible steps at the learning rate transitions.

---

### Step 12: Results Interpretation

Expected outcomes:
- PINN matches erfc solution to within 1-2% across all three time snapshots
- Largest errors occur near x=0 at early times (the steep gradient region)
- The (1-x) structure guarantees perfect accuracy at the far boundary
- Diffusion penetration at t=3600s reaches roughly 0.3-0.4 mm into the 2mm domain

---

## Training Flow Diagram

```
+----------------------------------------------------------+
|                   INITIALIZATION                          |
|  Sobol(2D) -> 500 uniform + 500 biased interior points   |
|  Linspace -> 250 surface BC + 250 initial condition pts   |
+----------------------------------------------------------+
                           |
                           v
+----------------------------------------------------------+
|                   FORWARD PASS                            |
|                                                          |
|  Interior: [x, tau, sqrt(tau)] -> DiffusionNet -> C      |
|            output = (1-x) * sigmoid(raw)                 |
|                                                          |
|  Surface:  model(0, tau) -> C_surface (target: 1.0)      |
|  Initial:  model(x, 0)  -> C_initial (target: 0.0)      |
+----------------------------------------------------------+
                           |
                           v
+----------------------------------------------------------+
|                AUTOGRAD DERIVATIVES                       |
|                                                          |
|  C_tau  = dC/d(tau)      [temporal rate of change]       |
|  C_x    = dC/d(x*)       [concentration gradient]       |
|  C_xx   = d2C/d(x*)^2    [curvature / diffusion term]   |
+----------------------------------------------------------+
                           |
                           v
+----------------------------------------------------------+
|                   LOSS COMPUTATION                        |
|                                                          |
|  residual = C_tau - 0.009 * C_xx                         |
|  loss_pde = mean(residual^2)                             |
|  loss_surface = mean((C_surface - 1)^2)                  |
|  loss_initial = mean(C_initial^2)                        |
|                                                          |
|  total = loss_pde + 20*loss_surface + 20*loss_initial    |
+----------------------------------------------------------+
                           |
                           v
+----------------------------------------------------------+
|                BACKWARD + UPDATE                          |
|                                                          |
|  loss.backward() -> gradients through entire graph       |
|  Adam step (lr halves every 5000 epochs)                 |
|  Save state if loss < best_loss                          |
+----------------------------------------------------------+
                           |
                    [repeat 12000x]
                           |
                           v
+----------------------------------------------------------+
|                POST-TRAINING                              |
|                                                          |
|  Load best state -> evaluate at t=900,1800,3600s         |
|  Compare to erfc analytical solution                     |
|  Plot concentration profiles + convergence               |
+----------------------------------------------------------+
```

---

## Comparison to Other PINN Projects

| Feature | Radioactive Decay (ODE) | Carbon Diffusion (PDE) |
|---------|------------------------|----------------------|
| Equation type | ODE (one variable: t) | PDE (two variables: x, t) |
| Independent vars | 1 (time only) | 2 (space and time) |
| Derivatives needed | dN/dt | dC/dt, dC/dx, d2C/dx2 |
| Autograd calls/step | 1 | 3 |
| Domain | 1D line | 2D rectangle |
| Sampling strategy | Linspace or Sobol 1D | Sobol 2D + biased |
| Boundary conditions | 1 (initial value) | 3 (surface + far + initial) |
| Hard constraints | exp(-x)*sigmoid (decay) | (1-x)*sigmoid (far BC) |
| Output bounds | [0, N0] | [0, 1] |
| Analytical solution | Exponential decay | erfc function |
| Training epochs | 5000-8000 | 12000 |
| Network inputs | 1 (t) | 3 (x, tau, sqrt(tau)) |
| Corner singularity | No | Yes (x=0, t=0) |
| Loss weights | Equal or mild | 20x on boundaries |
| Nondimensionalization | Optional | Essential |

---

## Key Physics Concepts

### Why PDE, Not ODE?

An ODE has one independent variable (typically time). You track how something evolves along a single axis. A PDE has multiple independent variables (here, both space and time). The concentration at any point depends on what is happening at neighboring points in space AND on the history in time.

This means:
- The network input is 2D (or 3D with the engineered sqrt feature)
- You need partial derivatives with respect to each variable separately
- The training domain is a rectangle [0,1] x [0,1], not just a line segment
- Boundary conditions exist on multiple edges of this rectangle

### The Similarity Variable and erfc

The quantity eta = x / (2*sqrt(D*t)) is special because the PDE solution depends only on this combination, not on x and t separately. This is called a "similarity solution." The erfc function emerges because:

1. Start with the diffusion equation
2. Substitute C(x,t) = f(eta) where eta = x/(2*sqrt(D*t))
3. The PDE reduces to an ODE: f'' + 2*eta*f' = 0
4. Solve with boundary conditions: f(0)=1, f(infinity)=0
5. The answer is f(eta) = erfc(eta)

The erfc function looks like a smoothed step: it equals 1 at eta=0 and decays to 0 as eta grows. The "width" of the transition zone grows as sqrt(t), which is the hallmark of diffusive processes.

### The Corner Singularity

At the corner (x=0, t=0), two incompatible conditions collide:
- Surface BC requires C(0, t>0) = 1
- Initial condition requires C(x>0, 0) = 0

No continuous function can satisfy both simultaneously at this single point. The mathematical solution has a discontinuity here. The PINN handles this by:
1. Never placing a training point exactly at (0,0)
2. Starting boundary points slightly away from the corner
3. Letting the network develop a steep but smooth transition

This is not a hack. It reflects the physical reality that the concentration at the surface rises extremely fast (essentially instantaneously on the timescale of the problem) but not literally at t=0.

### Why Sobol Beats Random and Grid

For training a PINN on a 2D domain:

**Random sampling** leaves gaps by chance. With 1000 points in [0,1]x[0,1], some regions might have 5 nearby points while others have none within 0.05 radius. The network can "cheat" by fitting well only where points cluster.

**Grid sampling** (e.g., 32x32 = 1024 points) is rigid. It cannot be easily adapted, and it scales poorly to higher dimensions (curse of dimensionality). In 2D it is acceptable but wasteful because many grid points land in the boring flat region.

**Sobol sampling** guarantees that after N points, every sub-region of the domain has received approximately its fair share. The discrepancy (worst-case deviation from uniformity) decreases as O(log(N)^d / N) compared to O(1/sqrt(N)) for random. For 1000 points in 2D, Sobol coverage is dramatically more uniform.

---

## Common Failure Modes

1. **Not biasing points toward the front:** Without the squared-Sobol trick, the network sees mostly flat C=0 regions and underresolves the steep diffusion front.

2. **Forgetting the tau shift:** Training with tau=0 points in the interior causes NaN gradients because the analytical solution has a singularity there.

3. **Insufficient boundary weight:** With equal weights, the PDE loss (1000 points) drowns out boundary losses (250 points each). The network satisfies the PDE approximately everywhere but gets the boundaries wrong, producing a physically meaningless solution.

4. **No sqrt(tau) feature:** The network can still converge, but requires 2-3x more epochs because it must implicitly learn the sqrt relationship through hidden layer combinations.

5. **Using ReLU instead of tanh:** ReLU has zero second derivative almost everywhere. Since the PDE involves C_xx, ReLU networks produce identically zero diffusion terms in large regions, making training very difficult.

---

## Running the Code

```bash
python carbon_diffusion.py
```

Expected output:
- Training progress printed every 1000 epochs (loss values for PDE, surface, initial)
- Final evaluation errors at t=900s, 1800s, 3600s (should be < 2% max error)
- Two-panel figure saved showing concentration profiles overlaid with analytical solution

Typical training time: 2-4 minutes on CPU, under 1 minute on GPU.
