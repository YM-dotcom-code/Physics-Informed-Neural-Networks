# **Pressure-Driven Channel Flow Physics-Informed Neural Network**

This project implements a **Physics-Informed Neural Network (PINN)** that solves the steady-state Poiseuille flow problem - viscous fluid driven by a pressure gradient between two stationary parallel plates.

The neural network predicts the velocity field `u(y)` across the channel by enforcing the momentum equation using PyTorch automatic differentiation. No training data is needed. The governing equation itself supervises the learning.

<p align="center">
  <img width="1834" height="1423" alt="Stokes Flow Diagram" src="https://github.com/user-attachments/assets/5739bd1c-11f8-43ca-a705-8cbb97eab5b5" />

</p>

## **Table of Contents**

- [Background](#background)
- [Key Terminology](#key-terminology)
- [Physical Problem](#physical-problem)
- [Governing Equation](#governing-equation)
- [Exact Analytical Solution (Step by Step)](#exact-analytical-solution-step-by-step)
- [PINN Implementation Workflow](#pinn-implementation-workflow)
- [Network Architecture](#network-architecture)
- [Hard Boundary Constraints](#hard-boundary-constraints)
- [Physics Residual and Loss Function](#physics-residual-and-loss-function)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [Comparison to Thermoelastic Bar Project](#comparison-to-thermoelastic-bar-project)
- [Assumptions and Limitations](#assumptions-and-limitations)
- [Possible Extensions](#possible-extensions)
- [Requirements](#requirements)
- [Run the Project](#run-the-project)
- [Code Walkthrough](CODE_WALKTHROUGH.md)

---

## **Background**

### What Is a Physics-Informed Neural Network (PINN)?

Traditional numerical methods like the Finite Element Method (FEM) solve differential equations by breaking the domain into a mesh and solving algebraic equations at each node. This works well but requires mesh generation, can be expensive, and needs re-running for each new set of parameters.

A PINN takes a different approach: it uses a neural network to approximate the solution and enforces the physics through the loss function. Instead of training on data, the network learns by minimizing how badly it violates the governing equation at sampled points. The physics equation itself acts as the teacher.

### What Is Poiseuille (Channel) Flow?

Poiseuille flow is one of the most fundamental solutions in fluid mechanics. It describes what happens when a viscous fluid is pushed through a channel by a pressure difference: the fluid moves fastest in the center and slows to zero at the walls. This creates the characteristic **parabolic velocity profile** seen in pipes, blood vessels, and microfluidic devices.

### What Is an ODE?

An **ODE (Ordinary Differential Equation)** is an equation that relates a function to its derivatives with respect to a single variable. "Ordinary" means there is only one independent variable (in our case, the position y across the channel). This is different from a **PDE (Partial Differential Equation)**, which involves derivatives with respect to multiple variables (like both position and time).

The equation governing this problem, `u''(y) = -2`, is an ODE because velocity u depends only on y.

### What Are the Navier-Stokes Equations?

The **Navier-Stokes equations** are the fundamental equations that govern all fluid flow - from ocean currents to airflow over a wing. They are a system of PDEs that relate velocity, pressure, and viscosity. In most cases they are too complex to solve by hand and require numerical methods.

For our problem (steady, 1D, incompressible, no gravity), the full Navier-Stokes system simplifies down to a single, solvable ODE. This makes it a perfect PINN benchmark.

### Why Solve This With a PINN?

This problem has an exact analytical solution, making it ideal for testing whether a PINN can:

- Learn a smooth parabolic velocity profile from a single ODE
- Enforce no-slip wall boundary conditions via hard constraints
- Converge to machine-precision accuracy without any training data

It is a stepping stone between basic 1D problems (like the coupled thermoelastic bar) and more complex fluid dynamics.

---

## **Key Terminology**

| Term | Meaning |
|------|---------|
| **Viscosity (mu)** | A measure of how "thick" or resistant to flow a fluid is. Water has low viscosity; honey has high viscosity. Units: Pa*s |
| **Pressure gradient (dp/dx)** | How quickly pressure changes along the flow direction. A negative value means pressure decreases downstream, pushing the fluid forward |
| **No-slip condition** | At a solid wall, fluid velocity equals zero. The fluid "sticks" to the surface. This is an experimentally observed fact for all viscous fluids |
| **Parabolic profile** | The velocity distribution has the shape of a parabola (u = 1-y^2). Maximum at the center, zero at the walls |
| **Boundary condition (BC)** | A known value at the edge of the domain. Here: u = 0 at both walls |
| **Hard constraint** | A boundary condition built directly into the network's output formula so it is impossible to violate |
| **Collocation points** | Specific locations in the domain where we check whether the equation is satisfied during training |
| **Residual** | The amount by which the network's prediction violates the governing equation. If the residual is zero, the physics is perfectly satisfied |
| **Autograd** | PyTorch's automatic differentiation engine. It computes exact derivatives of the network output - not approximations |
| **Epoch** | One complete pass through all training points. The network updates its weights once per epoch |
| **Loss function** | A single number that measures "how wrong is the network right now?" Training minimizes this |

---

## **Physical Problem**

### What We Are Solving

Imagine two infinite, flat, parallel plates that are stationary (not moving). The gap between them is filled with a viscous fluid (think of honey or oil). Now apply a pressure difference - higher pressure on one side, lower on the other. The fluid starts flowing through the channel, driven by this pressure gradient.

**Key physical behaviors:**
- The fluid sticks to both walls (zero velocity at the surface) - this is the no-slip condition
- The fluid moves fastest at the center of the channel where it is farthest from both walls
- The velocity profile is symmetric about the centerline
- The resulting shape is a parabola - this is not an assumption, it falls directly out of the math

### The Setup

- **Geometry**: Two infinite, parallel, stationary plates separated by a gap of 2h (total channel width = 2 meters)
- **Fluid**: Incompressible, Newtonian with dynamic viscosity mu = 1.0 Pa*s
- **Driving force**: A constant pressure gradient dp/dx = -2.0 Pa/m pushes the fluid in the positive x-direction
- **Domain**: y in [-h, +h] = [-1, +1] meters (vertical coordinate across the channel)
- **What we want to find**: The velocity u(y) at every point across the channel

### Parameters

| Parameter | Symbol | Value | Unit | Physical Meaning |
|-----------|--------|-------|------|-----------------|
| Dynamic viscosity | mu | 1.0 | Pa*s | How thick/resistant the fluid is |
| Pressure gradient | dp/dx | -2.0 | Pa/m | Driving force (negative = flow goes right) |
| Half-channel height | h | 1.0 | m | Distance from center to wall |

### Boundary Conditions

| Location | Condition | Physical Meaning |
|----------|-----------|-----------------|
| y = -h = -1 m | u(-h) = 0 | No-slip: fluid sticks to bottom wall |
| y = +h = +1 m | u(+h) = 0 | No-slip: fluid sticks to top wall |

---

## **Governing Equation**

The Navier-Stokes equations for steady, fully-developed, incompressible flow between parallel plates simplify to a single ODE:

```
mu * u''(y) = dp/dx
```

Where:
- `u(y)` = velocity in the flow direction at height y (what we want to find)
- `u''(y)` = second derivative of velocity with respect to y. The prime notation (') means "derivative": u' = du/dy (slope of the profile), u'' = d^2u/dy^2 (curvature of the profile)
- `mu` = dynamic viscosity (the fluid's resistance to flow)
- `dp/dx` = pressure gradient (the force pushing the fluid)

**In words:** The curvature (how much the velocity profile "bends") is constant and equals the pressure gradient divided by viscosity. A stronger pressure gradient creates faster flow. A more viscous fluid flows slower for the same pressure.

**Physical intuition:** Viscosity acts like friction between fluid layers. The pressure gradient pushes the fluid forward, while viscosity resists that motion. The balance between these two forces creates the parabolic profile - the center is far from both walls (less friction), so it moves fastest.

---

## **Exact Analytical Solution (Step by Step)**

Since this problem has an exact closed-form solution, we solve it by hand first. This gives us the "ground truth" to verify the PINN against.

### Step 1: Start with the ODE

```
mu * u''(y) = dp/dx
```

Divide both sides by mu:

```
u''(y) = dp/dx / mu = -2 / 1 = -2
```

### Step 2: Integrate once to get u'(y)

```
u'(y) = -2y + C1
```

### Step 3: Integrate again to get u(y)

```
u(y) = -y^2 + C1*y + C2
```

### Step 4: Apply boundary condition u(-1) = 0 (bottom wall)

```
0 = -(-1)^2 + C1*(-1) + C2
0 = -1 - C1 + C2
C2 = 1 + C1     ... (i)
```

### Step 5: Apply boundary condition u(+1) = 0 (top wall)

```
0 = -(1)^2 + C1*(1) + C2
0 = -1 + C1 + C2
C2 = 1 - C1     ... (ii)
```

### Step 6: Solve for constants

From (i) and (ii): `1 + C1 = 1 - C1` gives `2*C1 = 0`, so **C1 = 0**

Substituting back: **C2 = 1**

### Final Solution

```
u(y) = 1 - y^2     (m/s)
```

Or in general form (valid for any mu, dp/dx, h):

```
u(y) = -(dp/dx) * (h^2 - y^2) / (2*mu)
```

### Numerical Verification

| y position | u(y) = 1 - y^2 | Physical location |
|-----------|---------------|-------------------|
| y = -1.0 | 0 m/s | Bottom wall (no-slip satisfied) |
| y = -0.5 | 0.75 m/s | Quarter height |
| y = 0.0 | **1.0 m/s** | Centerline (maximum velocity) |
| y = +0.5 | 0.75 m/s | Three-quarter height |
| y = +1.0 | 0 m/s | Top wall (no-slip satisfied) |

### Verify the Solution Satisfies the PDE

```
u(y)  = 1 - y^2
u'(y) = -2y
u''(y) = -2

mu * u'' = 1.0 * (-2) = -2 = dp/dx    PDE satisfied everywhere
```

---

## **PINN Implementation Workflow**

The following steps describe how we solve this problem with a PINN. This is the same general workflow used in the [Coupled Thermoelastic Bar](../Coupled_Thermoelastic_Bar/) project.

```text
Step 1: Define the PDE
        mu*u''(y) = dp/dx, with u(-h) = u(+h) = 0
                           |
                           v
Step 2: Solve analytically (for validation)
        u(y) = 1 - y^2
                           |
                           v
Step 3: Design the network
        3x30 tanh, single output, input normalized (s = y/h)
                           |
                           v
Step 4: Enforce boundary conditions
        Hard: u = scale * (1-s^2) * N(s) -> u(+-h) = 0 always
                           |
                           v
Step 5: Sample collocation points
        80 uniform points in [-1, +1]
                           |
                           v
Step 6: Compute PDE residual via autograd
        r = u'' - (dp/dx)/mu   (should be 0)
                           |
                           v
Step 7: Assemble loss function
        loss = mean( (r / scale)^2 )   single term, no balancing!
                           |
                           v
Step 8: Train with Adam + StepLR
        10,000 epochs, save best model state
                           |
                           v
Step 9: Validate
        Compare to analytical solution on 200 test points
```

---

## **Network Architecture**

```
Input: y (position across channel)
    |
    v  normalize: s = y / h         (maps [-1, +1] to [-1, +1])
    |
    +-- Linear(1 -> 30) + Tanh
    +-- Linear(30 -> 30) + Tanh
    +-- Linear(30 -> 30) + Tanh
    +-- Linear(30 -> 1)  ->  raw_velocity
    |
    v  hard constraint: u = velocity_scale * (1 - s^2) * raw_velocity
    |
Output: u(y) (velocity at position y)
```

| Component | Choice | Reason |
|-----------|--------|--------|
| Hidden layers | 3 x 30 neurons | Sufficient for a smooth parabolic solution |
| Activation | Tanh | Smooth and infinitely differentiable (needed for computing u'' via autograd) |
| Input normalization | s = y/h | Maps domain to [-1, +1] for stable training |
| Output transform | (1-s^2) * N(s) | Hard-codes both no-slip boundary conditions |

**Why Tanh?** We need to compute u'' (second derivative) during training. This requires the activation function to be smooth and twice-differentiable everywhere. Tanh satisfies this. ReLU would not work because it has a discontinuous derivative at zero.

---

## **Hard Boundary Constraints**

Both wall boundary conditions are enforced **exactly** through the network's output formula:

```python
u = velocity_scale * (1 - s^2) * raw_velocity
```

**How it works:**

| Location | s = y/h | (1 - s^2) | Result | BC satisfied? |
|----------|---------|-----------|--------|--------------|
| Bottom wall (y = -1) | -1 | 1 - 1 = 0 | u = 0 | Always |
| Centerline (y = 0) | 0 | 1 - 0 = 1 | u = scale * raw | Free to learn |
| Top wall (y = +1) | +1 | 1 - 1 = 0 | u = 0 | Always |

The factor `(1 - s^2)` is zero at both walls and positive everywhere in between. No matter what the network outputs as `raw_velocity`, the final velocity is always zero at the walls. The boundary conditions are mathematically impossible to violate.

**Consequences:**
- No boundary loss terms needed in the loss function
- The loss is purely the PDE residual - no balancing weights to tune
- Training focuses 100% on satisfying the physics equation
- Wall error is machine zero (~10^-16) from epoch 1

**Bonus insight:** The exact solution is `u = 1 - y^2 = (1 - s^2) * 1.0`. The hard constraint already has the correct shape. The network only needs to learn `raw_velocity = 1.0` (a constant), which makes convergence extremely fast.

---

## **Physics Residual and Loss Function**

### How the PINN Checks the Physics

At each of the 80 collocation points, the network:
1. Predicts velocity `u(y)` via a forward pass
2. Computes `u'(y)` and `u''(y)` using PyTorch autograd (exact derivatives through the computational graph)
3. Evaluates the PDE residual: `r(y) = u''(y) - (dp/dx)/mu`

If the network has learned the correct solution, `r(y) = 0` everywhere. Any nonzero residual means the physics is being violated at that point.

### Residual Normalization

The residual is divided by a characteristic scale before squaring:

```
residual_scale = |dp/dx / mu| = |-2 / 1| = 2.0
```

Why normalize? The raw residual has units (m/s^2). Its magnitude depends on the physical parameters. Dividing by the characteristic scale makes the loss dimensionless and keeps gradient magnitudes in a healthy range for the optimizer.

### The Loss Function

```
loss = (1/N) * sum( (r(y_i) / residual_scale)^2 )
```

Where N = 80 (number of collocation points).

This is a **single-term loss** - there are no boundary penalty terms because both BCs are hard-coded. The simplicity of having just one term means:
- No hyperparameter tuning for loss weights
- No competition between different objectives
- Stable, monotonic convergence
- The optimizer focuses all its effort on satisfying the PDE

### Autograd: The Key Ingredient

PyTorch's automatic differentiation computes **exact** derivatives of the network output with respect to its input. This is not finite differences (which would introduce approximation error). The derivatives are analytically correct through the entire computational graph, enabling the PINN to check the physics with the same precision as the forward pass.

---

## **Training Configuration**

| Setting | Value | Reason |
|---------|-------|--------|
| Collocation points | 80 (uniform in [-1, +1]) | More than enough for a smooth parabola |
| Epochs | 10,000 | Converges well before this |
| Optimizer | Adam (lr = 1e-3) | Adaptive learning rates, standard choice |
| Scheduler | StepLR (halve every 5,000 epochs) | Big steps early, fine-tune later |
| Best-state tracking | Yes | Keeps the best model, avoids late noise |
| Loss function | MSE of normalized PDE residual | Single term - no balancing needed |

---

## **Results**

<p align="center">
  <img width="2374" height="877" alt="image" src="https://github.com/user-attachments/assets/315817f2-bc0b-4d8f-ac44-33ec0c5e622f" />

</p>

| Metric | Value | Meaning |
|--------|-------|---------|
| Maximum absolute error | ~10^-5 to 10^-6 | Worst-case error at any point |
| Relative L2 error | ~10^-5 to 10^-6 | Overall error relative to solution magnitude |
| Wall velocity error | ~10^-16 | Machine zero - hard constraint guarantee |
| Centerline velocity (PINN) | ~1.000000 m/s | |
| Centerline velocity (exact) | 1.000000 m/s | |

**Key observations:**
- The wall error is machine epsilon because the boundaries are hard-coded, not learned
- The network achieves ~6 digits of accuracy with only a 3x30 architecture
- Convergence is fast because the hard constraint `(1-s^2)` already has the exact parabolic shape
- With only 1 loss term (no boundary penalties), training is trivial - no weight balancing needed
- The PINN and analytical curves are visually indistinguishable in the plot

---

## **Comparison to Thermoelastic Bar Project**

This project and the [Coupled Thermoelastic Bar](../Coupled_Thermoelastic_Bar/) both use PINNs with hard boundary constraints, normalized residuals, and best-state tracking. Here is how they differ:

| Aspect | Thermoelastic Bar | Channel Flow |
|--------|-------------------|--------------|
| Physics domain | Solid mechanics + heat transfer | Fluid mechanics |
| Number of PDEs | 2 (coupled) | 1 (single ODE) |
| Unknowns | Temperature T(x) and displacement u(x) | Velocity u(y) |
| Architecture | 4 layers x 64 neurons, two output heads | 3 layers x 30 neurons, single output |
| Hard BCs | 3 (T at both ends, u at fixed end) | 2 (velocity at both walls) |
| Soft BCs | 1 (stress-free at free end, weight=100) | 0 (none needed!) |
| Loss terms | 3 (heat residual + mechanics residual + BC penalty) | 1 (PDE residual only) |
| Training epochs | 15,000 | 10,000 |
| Coupling | One-way: T drives u | None (single equation) |
| Expected accuracy | ~10^-4 | ~10^-6 |
| Main challenge | Balancing 3 loss terms + coupling | Almost none - trivial problem for PINN |

**Key takeaway:** The channel flow problem demonstrates that when all BCs can be hard-coded and there is only one equation, PINN training becomes trivial - no balancing, no tuning, fast convergence to high accuracy.

---

## **Assumptions and Limitations**

### Assumptions Made

| Assumption | Meaning | Impact on Solution |
|---|---|---|
| Steady state | No time dependence (flow is constant) | Reduces Navier-Stokes to a single ODE |
| Fully developed flow | No change in flow direction | Velocity depends only on y |
| Incompressible fluid | Constant density | Continuity equation is trivially satisfied |
| Newtonian fluid | Linear stress-strain relationship | Viscosity mu is constant |
| No body forces | No gravity effects | Eliminates gravity term |
| Infinite plates | No edge effects | 1D problem in y only |
| Laminar flow | Low Reynolds number | No turbulence modeling needed |

### Limitations of This PINN Implementation

- **1D only**: Cannot handle 2D/3D flow fields without architecture changes
- **Fixed parameters**: Changing mu, dp/dx, or h requires retraining from scratch
- **Smooth solutions only**: PINNs struggle with discontinuities or sharp gradients (not an issue here since the solution is a smooth parabola)
- **No turbulence**: Only valid for laminar flow (Re < ~2300 for channel flow)
- **Single phase**: Does not handle multiphase or non-Newtonian fluids

---

## **Possible Extensions**

| Extension | Difficulty | What Changes |
|-----------|-----------|-------------|
| Moving walls (Couette flow) | Easy | Change BCs: u(+h) = U_wall |
| Combined Couette-Poiseuille | Easy | Superposition of pressure-driven + wall-driven |
| Unsteady (start-up flow) | Moderate | Add time as input, PDE becomes parabolic |
| Non-Newtonian fluid | Moderate | Viscosity becomes a function of strain rate |
| 2D channel (entry region) | Hard | Add x-coordinate, need more complex architecture |
| Turbulent flow (RANS) | Hard | Add turbulence model equations |
| Parametric (varying dp/dx) | Moderate | Add dp/dx as a network input |
| Inverse problem | Moderate | Given velocity measurements, infer unknown mu or dp/dx |

### Suggested Next Steps (Increasing Complexity)

```
1. Pressure-Driven Channel Flow    <-- YOU ARE HERE
         |
         v
2. Couette-Poiseuille Flow          (add moving wall boundary condition)
         |
         v
3. Unsteady Start-Up Flow           (add time dimension)
         |
         v
4. 2D Navier-Stokes (Lid-Driven)   (full 2D, nonlinear)
```

---

## **Requirements**

```
torch >= 2.0
numpy
matplotlib
```

---

## **Run the Project**

```bash
python pressure_driven_channel_flow.py
```

**Output:**
- Console: training progress every 2,000 epochs + final accuracy metrics
- Image: `chapter4_stokes_flow.png` (velocity profile comparison + training convergence)

---

## **Code Walkthrough**

For a detailed line-by-line explanation of every section of the implementation, see [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md).
