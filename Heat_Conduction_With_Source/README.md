# Heat Conduction with Internal Generation: Physics-Informed Neural Network

A physics-informed neural network that solves steady one-dimensional heat conduction in a rod with uniform internal heat generation. This project deliberately uses **soft boundary constraints** with two different penalty weights to demonstrate how the choice of boundary penalty strength affects PINN accuracy. It serves as a teaching study on the penalty method in PINNs.



<img width="2223" height="1393" alt="heat_conduction_diagram" src="https://github.com/user-attachments/assets/35f92078-fac3-472f-b355-dd6c46c7963b" />

---

## Table of Contents

1. [Background](#background)
2. [Key Terminology](#key-terminology)
3. [Physical Problem](#physical-problem)
4. [Governing Equation](#governing-equation)
5. [Exact Analytical Solution](#exact-analytical-solution)
6. [PINN Implementation Workflow](#pinn-implementation-workflow)
7. [Network Architecture](#network-architecture)
8. [Soft Boundary Constraints](#soft-boundary-constraints)
9. [Physics Residual and Loss Function](#physics-residual-and-loss-function)
10. [Training Configuration](#training-configuration)
11. [Results](#results)
12. [Comparison to Other Projects](#comparison-to-other-projects)
13. [Assumptions and Limitations](#assumptions-and-limitations)
14. [Possible Extensions](#possible-extensions)
15. [Requirements](#requirements)
16. [How to Run](#how-to-run)
17. [Code Walkthrough](#code-walkthrough)

---

## Background

### What Is Heat Conduction with Internal Generation?

Heat conduction is the transfer of thermal energy through a material from regions of higher temperature to regions of lower temperature. In many practical situations, heat is also generated inside the material itself. Examples include:

- Electrical resistive heating in a wire carrying current
- Nuclear fuel rods where fission reactions release energy throughout the volume
- Concrete curing where chemical reactions produce heat internally
- Biological tissue with metabolic heat production

When heat is generated uniformly throughout a solid, the temperature distribution is no longer linear (as it would be for pure conduction without generation). Instead, the temperature profile becomes parabolic. The internally generated heat must flow outward through the boundaries, creating a temperature peak somewhere inside the domain.

### How This Differs from the Thermoelastic Heat Equation

The thermoelastic bar project in this collection solves a heat equation with NO internal generation. In that case, the steady-state temperature is a simple linear interpolation between the two boundary values. Adding internal heat generation introduces a source term that makes the governing equation a Poisson equation rather than a Laplace equation. The solution gains a quadratic component, and the physics residual must account for the source term.

### Why Soft Constraints for This Project?

Most other PINN projects in this collection use hard boundary constraints, where the network output is algebraically constructed to satisfy boundary conditions exactly. This project deliberately avoids hard constraints. Instead, it enforces boundary conditions through penalty terms in the loss function. The purpose is pedagogical: by training two models with different penalty weights (lambda_bc = 1 versus lambda_bc = 100), we can directly observe how penalty strength affects the network's ability to simultaneously satisfy the physics and the boundary conditions.

---

## Key Terminology

**Internal heat generation**: Thermal energy produced per unit volume per unit time within a material, measured in watts per cubic meter (W/m^3). Denoted Q in this project.

**Thermal conductivity**: A material property that quantifies how easily heat flows through the material, measured in watts per meter per kelvin (W/(m*K)). Denoted k. Higher k means the material conducts heat more readily.

**Dirichlet boundary condition**: A boundary condition that specifies the value of the solution (temperature) at a boundary point. Both boundaries in this problem have prescribed temperatures.

**Soft constraint**: A method of enforcing boundary conditions by adding a penalty term to the loss function. The network output is not forced to satisfy the boundary condition exactly. Instead, deviations from the prescribed boundary values contribute to the total loss, and the optimizer tries to minimize those deviations along with the physics residual.

**Hard constraint**: A method of enforcing boundary conditions by algebraically constructing the network output so that it automatically satisfies the boundary conditions for any set of network weights. Hard constraints guarantee exact boundary satisfaction but are not always easy to construct for complex geometries or boundary types.

**Penalty weight (lambda_bc)**: A scalar multiplier applied to the boundary condition loss term. A larger penalty weight tells the optimizer to prioritize satisfying the boundary conditions over minimizing the interior physics residual. Too small a weight allows boundary violations; too large a weight can slow convergence of the interior solution.

**Nondimensionalization**: The process of rescaling variables so that they become dimensionless (no units) and typically range between 0 and 1. This improves neural network training by keeping all quantities at similar magnitudes and preventing numerical issues from very large or very small numbers.

**Collocation points**: The discrete set of interior points where the physics residual (governing PDE) is evaluated during training. The network must satisfy the differential equation at these points.

**Physics residual**: The amount by which the network's predicted solution fails to satisfy the governing differential equation. A perfect solution would have zero residual everywhere.

**Poisson equation**: A second-order partial differential equation of the form nabla^2(u) = f, where f is a known source term. The steady heat equation with generation is a one-dimensional Poisson equation.

---

## Physical Problem

Consider a solid rod of length L = 1.0 meter with uniform internal heat generation. Both ends of the rod are held at fixed (but different) temperatures. We want to find the steady-state temperature distribution along the rod.

### Physical Setup

```
    Q (uniform internal heat generation throughout)
    |     |     |     |     |     |     |     |
    v     v     v     v     v     v     v     v
+---+---+---+---+---+---+---+---+---+---+---+---+
|                                                 |
|   T(0) = 100 C          Rod           T(L) = 200 C
|                                                 |
+---+---+---+---+---+---+---+---+---+---+---+---+
x=0                                             x=L

Heat flows LEFT from the peak    Heat flows RIGHT from the peak
toward the cooler left end       toward the warmer right end

```

### Parameters

| Parameter | Symbol | Value | Units |
| --- | --- | --- | --- |
| Rod length | L | 1.0 | m |
| Thermal conductivity | k | 50 | W/(m*K) |
| Internal heat generation rate | Q | 1000 | W/m^3 |
| Left boundary temperature | T(0) | 100 | degrees C |
| Right boundary temperature | T(L) | 200 | degrees C |

### Boundary Conditions

Both boundaries have Dirichlet (prescribed temperature) conditions:

- Left end (x = 0): T = 100 degrees C
- Right end (x = L): T = 200 degrees C

### Physical Intuition

The internal heat generation creates thermal energy throughout the rod. This energy must flow out through the two ends. Because the right end is warmer than the left end, more heat flows leftward toward the cooler boundary. The temperature reaches a maximum somewhere inside the rod (not at the midpoint, because the asymmetric boundary temperatures shift the peak toward the warmer end).

---

## Governing Equation

### In Plain English

At steady state, the temperature at every point in the rod is constant in time. This means that the net heat flowing into any small segment of the rod (from conduction through its faces) plus the heat generated inside that segment must equal zero. If more heat is generated than conducted away, the temperature would rise, violating the steady-state assumption.

### Mathematical Form

The one-dimensional steady-state heat equation with uniform internal generation is:

```
k * d^2T/dx^2 + Q = 0

```

where:

- k = thermal conductivity (W/(m*K))
- T = temperature (degrees C)
- x = position along the rod (m)
- Q = volumetric heat generation rate (W/m^3)
- d^2T/dx^2 = second spatial derivative of temperature (curvature of T profile)

Rearranging:

```
d^2T/dx^2 = -Q/k

```

This tells us the temperature profile has constant negative curvature (it curves downward like an inverted parabola) because Q and k are both positive. The magnitude of curvature is Q/k = 1000/50 = 20 degrees C per meter squared.

---

## Exact Analytical Solution

### Derivation (Step by Step)

**Step 1: Integrate the governing equation once.**

Starting from:

```
d^2T/dx^2 = -Q/k

```

Integrate with respect to x:

```
dT/dx = -(Q/k)*x + C1

```

where C1 is the first integration constant.

**Step 2: Integrate again.**

```
T(x) = -(Q/(2k))*x^2 + C1*x + C2

```

where C2 is the second integration constant.

**Step 3: Apply the left boundary condition T(0) = T0 = 100.**

```
T(0) = -(Q/(2k))*(0)^2 + C1*(0) + C2 = C2 = 100

```

So C2 = T0 = 100.

**Step 4: Apply the right boundary condition T(L) = TL = 200.**

```
T(L) = -(Q/(2k))*L^2 + C1*L + T0 = 200

```

Solve for C1:

```
C1*L = TL - T0 + (Q/(2k))*L^2
C1 = (TL - T0)/L + Q*L/(2k)

```

Substituting values:

```
C1 = (200 - 100)/1.0 + 1000*1.0/(2*50)
C1 = 100 + 10 = 110

```

**Step 5: Write the complete solution.**

```
T(x) = -(Q/(2k))*x^2 + C1*x + T0
T(x) = -10*x^2 + 110*x + 100

```

### Verification

- T(0) = -10*(0) + 110*(0) + 100 = 100 (matches left BC)
- T(1) = -10*(1) + 110*(1) + 100 = -10 + 110 + 100 = 200 (matches right BC)
- T''(x) = -20, and k*T'' + Q = 50*(-20) + 1000 = -1000 + 1000 = 0 (satisfies PDE)

### Location of Maximum Temperature

Setting dT/dx = 0:

```
dT/dx = -20*x + 110 = 0
x_max = 110/20 = 5.5 m

```

Since x_max = 5.5 m is outside the domain [0, 1], the temperature is monotonically increasing on [0, 1]. The maximum temperature on the domain occurs at x = L = 1.0 m, which is the right boundary: T(1) = 200 degrees C. The parabolic curvature bends the profile slightly below what a straight line would give, but the generation term (which adds only 10 degrees of curvature effect over this short domain) is not strong enough to create an interior maximum.

---

## PINN Implementation Workflow

### Nine-Step Process

```
Step 1: Define physical parameters
        k=50, Q=1000, L=1.0, T0=100, TL=200, T_scale=100
            |
            v
Step 2: Nondimensionalize the problem
        s = x/L in [0,1], theta = (T - T0)/T_scale
        PDE becomes: theta''(s) + source = 0, source = Q*L^2/(k*T_scale) = 0.2
            |
            v
Step 3: Define boundary conditions in nondimensional form
        theta(0) = (T0 - T0)/T_scale = 0
        theta(1) = (TL - T0)/T_scale = 1
            |
            v
Step 4: Build HeatNet (3 hidden layers, 30 neurons, tanh)
        Raw output with NO hard constraint
            |
            v
Step 5: Sample 80 collocation points in [0,1]
        Plus 2 boundary points at s=0 and s=1
            |
            v
Step 6: Compute physics residual at collocation points
        R_pde = d^2(theta)/ds^2 + source
            |
            v
Step 7: Compute boundary condition residuals
        R_bc = [theta(0) - 0, theta(1) - 1]
            |
            v
Step 8: Form total loss = mean(R_pde^2) + lambda_bc * mean(R_bc^2)
        Train with Adam optimizer, StepLR scheduler
            |
            v
Step 9: After training, convert predictions to physical units
        T(x) = theta(s) * T_scale + T0
        Evaluate physical residual: k*T''(x) + Q
        Plot results and compare both lambda values

```

### Comparison: Soft vs Hard Constraints

| Aspect | Soft Constraint (This Project) | Hard Constraint (Other Projects) |
| --- | --- | --- |
| BC satisfaction | Approximate, depends on lambda | Exact by construction |
| Network output | Raw, unconstrained | Algebraically modified |
| Loss function | PDE loss + weighted BC loss | PDE loss only |
| Hyperparameter | lambda_bc must be tuned | No BC weight needed |
| Flexibility | Works for any BC type | Requires custom construction |
| Teaching value | Shows trade-offs explicitly | Hides BC enforcement details |
| Risk | BCs may be poorly satisfied | None for BC accuracy |

---

## Network Architecture

### HeatNet Structure

```
Input: s (nondimensional position, scalar)
    |
    v
[Linear: 1 -> 30] --> [tanh]     Hidden Layer 1
    |
    v
[Linear: 30 -> 30] --> [tanh]    Hidden Layer 2
    |
    v
[Linear: 30 -> 30] --> [tanh]    Hidden Layer 3
    |
    v
[Linear: 30 -> 1]                Output Layer
    |
    v
Output: theta_raw (nondimensional temperature, scalar)

```

### Why No Hard Constraint?

In the other PINN projects in this collection (thermoelastic bar, channel flow, beam bending), the network output is modified algebraically so that boundary conditions are satisfied exactly regardless of what the network weights produce. For example, a typical hard constraint might look like:

```
theta_hard(s) = (1-s)*theta_left + s*theta_right + s*(1-s)*NN(s)

```

This project deliberately avoids such construction. The raw network output is used directly as the predicted nondimensional temperature. Boundary conditions are enforced only through the loss function. This choice is intentional: the goal is to demonstrate what happens when the boundary penalty weight is too weak versus appropriately strong.

### Parameter Count

- Layer 1: 1*30 weights + 30 biases = 60
- Layer 2: 30*30 weights + 30 biases = 930
- Layer 3: 30*30 weights + 30 biases = 930
- Output: 30*1 weights + 1 bias = 31
- Total: 1951 trainable parameters

---

## Soft Boundary Constraints

### The Penalty Method

In the penalty method for boundary condition enforcement, deviations from prescribed boundary values are penalized in the loss function. The network is free to predict any value at the boundaries, but predictions that violate the boundary conditions increase the total loss. The optimizer then tries to find weights that simultaneously minimize both the interior physics residual and the boundary violations.

### Mathematical Formulation

The boundary condition loss is:

```
Loss_bc = (1/N_bc) * sum( (theta_predicted(s_i) - theta_prescribed(s_i))^2 )

```

where s_i are the boundary points (s=0 and s=1), and theta_prescribed are the target values (0 and 1 respectively).

This loss is multiplied by a penalty weight lambda_bc before being added to the physics loss.

### Why the Weight Matters

**With lambda_bc = 1 (weak penalty):** The boundary loss and the physics loss are weighted equally. The optimizer treats boundary accuracy and interior physics accuracy as equally important. Since there are many more collocation points (80) than boundary points (2), the physics loss dominates the gradient signal. The network may converge to a solution that satisfies the PDE well in the interior but shows noticeable errors at the boundaries.

**With lambda_bc = 100 (strong penalty):** The boundary loss is amplified by a factor of 100. Even small deviations at the boundaries produce large loss contributions. The optimizer is strongly motivated to satisfy the boundary conditions, and the resulting solution matches the prescribed temperatures at x=0 and x=L much more accurately. The interior physics is still enforced, but now the boundaries are treated as high-priority constraints.

### The Trade-off

Increasing lambda_bc too much can cause problems:

- The optimizer may focus entirely on boundaries and neglect interior physics
- Gradient magnitudes from the BC loss may overwhelm PDE gradients
- Training may become unstable or oscillatory

The purpose of this project is to show that lambda_bc = 1 is too weak (boundaries are violated), while lambda_bc = 100 provides a good balance for this particular problem.

---

## Physics Residual and Loss Function

### Nondimensional PDE Residual

The governing equation in nondimensional form is:

```
d^2(theta)/ds^2 + source = 0

```

where source = Q*L^2 / (k*T_scale) = 1000 * 1.0^2 / (50 * 100) = 0.2.

The physics residual at each collocation point is:

```
R_pde(s) = d^2(theta)/ds^2 + 0.2

```

The second derivative is computed using PyTorch automatic differentiation (autograd). The input s requires gradients, and two successive differentiations with respect to s yield the second derivative.

### Total Loss Function

```
Loss_total = Loss_pde + lambda_bc * Loss_bc

```

where:

```
Loss_pde = (1/N_col) * sum( R_pde(s_j)^2 )    for j = 1, ..., N_col (80 points)

Loss_bc  = (1/N_bc) * sum( R_bc(s_i)^2 )       for i = 1, ..., N_bc (2 points)

```

- N_col = 80 (number of interior collocation points)
- N_bc = 2 (number of boundary points)
- lambda_bc = 1 or 100 (the variable under study)

### Physical Residual (for Evaluation)

After training, the solution is converted back to physical units and the physical residual is computed:

```
Physical residual = k * d^2T/dx^2 + Q

```

This should be zero everywhere for a perfect solution. The magnitude of the physical residual indicates how well the network has learned the governing physics.

---

## Training Configuration

| Parameter | Value | Reason |
| --- | --- | --- |
| Collocation points | 80 | Sufficient for this smooth 1D problem |
| Epochs | 6000 | Enough for convergence with StepLR |
| Optimizer | Adam | Good default for PINNs, handles saddle points |
| Initial learning rate | 1e-3 | Standard starting point for Adam |
| Scheduler | StepLR | Simple and predictable decay |
| Step size | 3000 epochs | Halve learning rate at midpoint of training |
| Gamma (decay factor) | 0.5 | Halve the learning rate each step |
| Best-state tracking | Yes | Save weights with lowest total loss |
| Boundary penalty (run 1) | lambda_bc = 1 | Weak penalty (baseline) |
| Boundary penalty (run 2) | lambda_bc = 100 | Strong penalty (improved) |

### Training Function

The `train_heat_pinn()` function encapsulates the full training loop. It accepts the penalty weight lambda_bc as an argument and returns:

- The trained model (loaded with best-state weights)
- Loss history (total loss at each epoch)
- Final metrics (PDE loss, BC loss, total loss)

Both runs use identical network architecture, initialization seeds, collocation points, and optimizer settings. The only difference is the value of lambda_bc.

---

## Results



<img width="2975" height="877" alt="image" src="https://github.com/user-attachments/assets/c615ebea-10d8-4e81-8049-ab83fb6d67b2" />


### Three-Panel Comparison

The output figure contains three panels showing both runs (lambda_bc=1 in one color, lambda_bc=100 in another) alongside the exact analytical solution.

**Panel 1: Temperature Profile T(x)**

Shows the predicted temperature distribution along the rod in physical units (degrees C) compared to the exact parabolic solution T(x) = -10x^2 + 110x + 100.

- With lambda_bc = 1: The predicted curve may deviate noticeably from the exact solution, particularly near the boundaries where the prescribed temperatures are not accurately matched.
- With lambda_bc = 100: The predicted curve closely follows the exact solution, with accurate boundary values and correct interior curvature.

**Panel 2: Physical PDE Residual k*T''(x) + Q**

Shows how well the trained network satisfies the governing equation pointwise. The residual should be zero everywhere for a perfect solution.

- With lambda_bc = 1: The residual may be small in parts of the interior but shows nonzero values, especially near boundaries where the solution deviates.
- With lambda_bc = 100: The residual is uniformly small across the entire domain, indicating the network has learned the correct physics.

**Panel 3: Convergence History (Loss vs Epoch)**

Shows the total loss over 6000 epochs for both runs on the same axes.

- Both runs show initial rapid decrease as Adam finds the general solution shape.
- The lambda_bc = 100 run may show initially higher total loss (because BC violations are more heavily penalized) but converges to a more physically accurate solution.
- The StepLR scheduler creates a visible change in convergence rate around epoch 3000.

### Key Observations

1. **Boundary accuracy depends on penalty weight.** With lambda_bc = 1, the network does not strongly enforce T(0) = 100 and T(L) = 200. With lambda_bc = 100, boundary values are matched accurately.
2. **Interior physics is affected by boundary accuracy.** Because the analytical solution depends on both boundary conditions and the PDE, an inaccurate boundary effectively changes the problem the network is solving. Poor boundaries lead to poor interior solutions.
3. **The penalty method requires tuning.** Unlike hard constraints that guarantee exact boundary satisfaction, soft constraints introduce a hyperparameter (lambda_bc) that must be chosen carefully. This project demonstrates why many PINN practitioners prefer hard constraints when they are available.

---

## Comparison to Other Projects

| Feature | Heat Conduction (this) | Thermoelastic Bar | Channel Flow | Beam Bending |
| --- | --- | --- | --- | --- |
| Physics | Heat + generation | Heat (no source) | Fluid flow | Structural |
| PDE order | 2nd order | 2nd order | 2nd order | 4th order |
| Source term | Yes (Q = 1000) | No | Yes (pressure) | Yes (load) |
| BC type | Dirichlet only | Dirichlet only | Dirichlet | Mixed |
| Constraint method | Soft penalty | Hard | Hard | Hard |
| Hyperparameter study | Yes (lambda) | No | No | No |
| Number of models trained | 2 | 1 | 1 | 1 |
| Solution shape | Parabolic | Linear | Parabolic | Polynomial |
| Teaching focus | Penalty weight effect | Coupled multiphysics | Flow physics | Higher-order PDE |

### Relationship to the Thermoelastic Bar

The thermoelastic bar project solves the SAME type of equation (steady 1D heat conduction) but without internal generation (Q = 0). That simplification makes the exact solution a straight line between the boundary temperatures. Adding Q != 0 in this project introduces curvature (the parabolic term -10x^2) and makes the problem a Poisson equation rather than a Laplace equation.

---

## Assumptions and Limitations

### Physical Assumptions

1. **Steady state.** The temperature does not change with time. All transients have died out.
2. **One-dimensional.** Heat flows only along the rod axis. There is no radial heat loss (the rod is perfectly insulated on its lateral surface, or is very thin).
3. **Uniform generation.** Q is constant throughout the rod. In reality, heat generation may vary with position or temperature.
4. **Constant thermal conductivity.** k does not depend on temperature. For many materials, k varies with T, especially over large temperature ranges.
5. **No radiation or convection.** Heat transfer occurs only by conduction within the rod.

### Numerical Limitations

1. **Soft constraints do not guarantee boundary accuracy.** Even with lambda_bc = 100, the boundary conditions are satisfied only approximately. For problems requiring exact boundary satisfaction, hard constraints are preferable.
2. **Only two penalty weights tested.** A more thorough study would sweep lambda_bc over a wider range (0.1, 1, 10, 100, 1000, 10000) to map out the full behavior.
3. **Fixed architecture.** The 3x30 tanh network was not optimized for this problem. Different architectures might show different sensitivity to lambda_bc.
4. **Single random seed.** Results may vary with different initializations. A robust study would average over multiple seeds.
5. **Simple 1D problem.** The conclusions about penalty weights extend qualitatively to higher dimensions and more complex PDEs, but the specific optimal lambda_bc value is problem-dependent.

---

## Possible Extensions

1. **Penalty weight sweep.** Train models with lambda_bc in {0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000} and plot boundary error versus interior physics error as a Pareto front.
2. **Adaptive penalty weighting.** Implement schemes that automatically adjust lambda_bc during training based on the relative magnitudes of the PDE and BC losses (similar to learning rate annealing for loss balancing).
3. **Comparison with hard constraints.** Add a third model that uses hard constraints (the algebraic construction method) and compare its accuracy and convergence speed to the best soft-constraint model.
4. **Temperature-dependent conductivity.** Make k a function of T, turning the PDE into a nonlinear equation. The exact solution no longer exists in closed form, making the PINN approach more valuable.
5. **Nonuniform generation.** Let Q vary with position (Q(x) = Q0*sin(pi*x/L), for example) to create more complex temperature profiles.
6. **Mixed boundary conditions.** Replace one Dirichlet BC with a Neumann (prescribed heat flux) or Robin (convective) condition. Study how soft penalty enforcement works for derivative boundary conditions.
7. **Time-dependent version.** Solve the transient heat equation with generation to observe how the temperature evolves from an initial condition to the steady-state parabolic profile.
8. **Two-dimensional extension.** Solve heat conduction with generation on a rectangular plate, comparing soft versus hard constraint enforcement in 2D.

---

## Requirements

```
torch >= 1.10
numpy >= 1.20
matplotlib >= 3.4

```

No GPU required. Training completes in under one minute on CPU for both runs combined.

---

## How to Run

```bash
cd Heat_Conduction_With_Source
python heat_conduction_with_source.py

```

The script will:

1. Train the first model with lambda_bc = 1 (weak penalty) for 6000 epochs
2. Train the second model with lambda_bc = 100 (strong penalty) for 6000 epochs
3. Convert both solutions to physical units
4. Compute physical PDE residuals for both models
5. Generate the three-panel comparison figure (chapter5_heat_conduction.png)
6. Print metrics for both runs (final losses, max boundary error, max residual)

---

## Code Walkthrough

For a detailed line-by-line explanation of the implementation, see [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md).

Key functions in `heat_conduction_with_source.py`:

- `HeatNet`: Neural network class (3 hidden layers, 30 neurons, tanh activation, raw output)
- `train_heat_pinn(lambda_bc)`: Full training loop with best-state tracking, returns trained model and loss history
- `compute_pde_residual(model, s)`: Evaluates d^2(theta)/ds^2 + source using autograd
- `evaluate_physical(model, x)`: Converts nondimensional predictions to physical temperature and computes k*T''+Q
- Main block: Runs both training configurations, generates comparison plots

