# Cantilever Beam Deflection - Physics-Informed Neural Network

A physics-informed neural network (PINN) that learns the deflection shape of a cantilever beam under uniform distributed load by embedding the Euler-Bernoulli beam equation directly into the training loss.

<p>
<img width="2218" height="1129" alt="beam_deflection_diagram" src="https://github.com/user-attachments/assets/a640a70a-22d3-4694-8a4b-f46308ca5ba7" />
</p>

---

## Table of Contents

1. [Background](#background)
2. [Key Terminology](#key-terminology)
3. [Physical Problem](#physical-problem)
4. [Governing Equation](#governing-equation)
5. [Exact Analytical Solution](#exact-analytical-solution)
6. [PINN Implementation Workflow](#pinn-implementation-workflow)
7. [Network Architecture](#network-architecture)
8. [Hard Boundary Constraints](#hard-boundary-constraints)
9. [Physics Residual and Loss Function](#physics-residual-and-loss-function)
10. [Training Configuration](#training-configuration)
11. [Results](#results)
12. [Comparison to Other Projects](#comparison-to-other-projects)
13. [Assumptions and Limitations](#assumptions-and-limitations)
14. [Possible Extensions](#possible-extensions)
15. [Requirements](#requirements)
16. [How to Run](#how-to-run)

---

## Background

### What is Euler-Bernoulli Beam Theory?

Euler-Bernoulli beam theory is the classical approach to predicting how beams bend under load. It assumes that cross-sections of the beam remain flat and perpendicular to the neutral axis during bending - meaning the beam deforms smoothly without warping or twisting. This simplification turns the complex 3D elasticity problem into a single ordinary differential equation (ODE) relating the beam's transverse deflection to the applied load.

The theory works well for slender beams (length much greater than depth) under small deflections - exactly the regime encountered in most structural engineering applications like floor joists, bridge girders, and machine components.

### What is a 4th-Order ODE?

The Euler-Bernoulli beam equation is a 4th-order ordinary differential equation. This means it involves the fourth derivative of the deflection with respect to position along the beam. Physically, each successive derivative carries a distinct mechanical meaning:

- 1st derivative (slope): the angle of the beam at each point
- 2nd derivative (curvature): proportional to the bending moment
- 3rd derivative: proportional to the shear force
- 4th derivative: proportional to the distributed load intensity

A 4th-order ODE requires exactly four boundary conditions to produce a unique solution - two at each end of the beam for a standard problem.

### What is a Cantilever?

A cantilever is a beam that is rigidly fixed at one end and completely free at the other. Think of a diving board bolted to the pool deck, a balcony extending from a building wall, or a shelf bracket attached to a wall. The fixed end cannot move or rotate, while the free end is unconstrained - it can deflect downward and rotate freely.

This configuration is structurally important because it produces the largest deflections and bending moments of any single-span beam support arrangement, making it a critical design case in engineering.

---

## Key Terminology

| Term | Definition |
|------|-----------|
| Deflection (w) | The transverse (vertical) displacement of the beam at any point along its length, measured from the unloaded straight position |
| Bending Moment (M) | The internal torque at a cross-section that causes the beam to curve; proportional to the second derivative of deflection: M = -EI * w''(x) |
| Shear Force (V) | The internal transverse force at a cross-section that resists sliding; proportional to the third derivative of deflection: V = -EI * w'''(x) |
| Bending Stiffness (EI) | The product of Young's modulus E (material stiffness) and second moment of area I (cross-section geometry); quantifies resistance to bending |
| Cantilever | A beam with one end rigidly clamped (zero displacement and zero slope) and the other end free (zero moment and zero shear) |
| Distributed Load (q) | A force spread uniformly along the beam length, measured in Newtons per meter (N/m); like the weight of snow on a roof beam |
| Dirichlet Boundary Condition | A condition that prescribes the value of the solution itself; here w(0) = 0 (fixed displacement) and w'(0) = 0 (fixed slope) |
| Neumann Boundary Condition | A condition that prescribes the value of a derivative of the solution; here w''(L) = 0 (zero moment) and w'''(L) = 0 (zero shear) |
| Nondimensionalization | Rescaling variables so they become unitless and of order 1; improves neural network training by keeping all quantities in a similar numerical range |

---

## Physical Problem

### What We Are Solving

We want to predict how much a cantilever beam bends when a uniform load is applied along its entire length. Imagine a horizontal steel beam bolted into a wall at the left end, with weight (such as its own self-weight or a layer of material) spread evenly along its span. The beam curves downward, with maximum deflection occurring at the free tip on the right end.

Given the beam's material properties, cross-section geometry, and the magnitude of the applied load, we want to find the deflection w(x) at every point x along the beam.

### Setup

- A beam of length L is clamped rigidly at x = 0 (the wall)
- The beam is free at x = L (the tip)
- A uniform distributed load q acts downward along the entire length
- We solve for the deflection curve w(x) from the fixed end to the free end

### Parameters

| Parameter | Symbol | Value | Units | Physical Meaning |
|-----------|--------|-------|-------|-----------------|
| Young's Modulus | E | 200 x 10^9 | Pa | Material stiffness (steel) |
| Second Moment of Area | I | 8.33 x 10^-6 | m^4 | Cross-section geometry factor |
| Bending Stiffness | EI | 1.666 x 10^6 | N*m^2 | Resistance to bending |
| Distributed Load | q | 1000 | N/m | Applied force per unit length |
| Beam Length | L | 1.0 | m | Span of the beam |
| Tip Deflection | w(L) | 0.075 | mm | Maximum deflection at free end |

### Boundary Conditions

| Location | Condition | Equation | Physical Meaning |
|----------|-----------|----------|-----------------|
| Fixed end (x=0) | Zero displacement | w(0) = 0 | The wall holds the beam in place - it cannot move vertically |
| Fixed end (x=0) | Zero slope | w'(0) = 0 | The wall prevents rotation - the beam leaves the wall horizontally |
| Free end (x=L) | Zero moment | w''(L) = 0 | No internal bending torque at the tip - nothing resists curvature there |
| Free end (x=L) | Zero shear | w'''(L) = 0 | No internal transverse force at the tip - nothing pushes the end vertically |

The fixed-end conditions (Dirichlet type) constrain the solution value and its first derivative. The free-end conditions (Neumann type) constrain the second and third derivatives. Together, these four conditions uniquely determine the solution to the 4th-order ODE.

---

## Governing Equation

The Euler-Bernoulli beam equation in its simplest form is:

```
EI * w''''(x) = q
```

In plain English: the bending stiffness (EI) times the fourth spatial derivative of the deflection (w'''') equals the applied distributed load (q) at every point along the beam.

### What the Fourth Derivative Means Physically

Each derivative of deflection has a physical interpretation built up from equilibrium of internal forces:

```
w(x)      -> deflection (how far the beam moves from straight)
w'(x)     -> slope (angle of the beam)
w''(x)    -> curvature, proportional to bending moment: M = EI * w''
w'''(x)   -> rate of change of moment, proportional to shear: V = EI * w'''
w''''(x)  -> rate of change of shear = applied load: q = EI * w''''
```

The beam equation states that wherever load is applied, the shear force must change - and since shear is the derivative of moment, and moment is the derivative of curvature, the equation connects load directly to the fourth derivative of deflection.

For our problem, q is constant everywhere (uniform load), so the fourth derivative of deflection is the same constant at every point. This makes the problem solvable by straightforward integration.

---

## Exact Analytical Solution

Because the load is constant, we can solve the ODE by integrating four times and applying the four boundary conditions. This provides the exact answer against which we validate the PINN.

### Step 1: Start with the governing equation

```
EI * w''''(x) = q
```

### Step 2: Integrate once (obtain shear force)

```
EI * w'''(x) = q*x + C1
```

Apply BC w'''(L) = 0 (zero shear at free end):

```
0 = q*L + C1  ->  C1 = -q*L
EI * w'''(x) = q*(x - L)
```

### Step 3: Integrate again (obtain bending moment)

```
EI * w''(x) = q*(x^2/2 - L*x) + C2
```

Apply BC w''(L) = 0 (zero moment at free end):

```
0 = q*(L^2/2 - L^2) + C2 = q*(-L^2/2) + C2  ->  C2 = q*L^2/2
EI * w''(x) = q*(x^2/2 - L*x + L^2/2) = (q/2)*(x - L)^2
```

### Step 4: Integrate again (obtain slope)

```
EI * w'(x) = (q/2)*(x^3/3 - L*x^2 + L^2*x) + C3
```

Apply BC w'(0) = 0 (zero slope at fixed end):

```
0 = 0 + C3  ->  C3 = 0
EI * w'(x) = (q/6)*(x^3 - 3*L*x^2 + 3*L^2*x)
```

### Step 5: Integrate once more (obtain deflection)

```
EI * w(x) = (q/6)*(x^4/4 - L*x^3 + (3/2)*L^2*x^2) + C4
```

Apply BC w(0) = 0 (zero displacement at fixed end):

```
0 = 0 + C4  ->  C4 = 0
```

### Final Exact Solution

```
w(x) = q * (x^4 - 4*L*x^3 + 6*L^2*x^2) / (24*EI)
```

### Verification: Tip Deflection

At x = L:

```
w(L) = q * (L^4 - 4*L^4 + 6*L^4) / (24*EI)
     = q * (3*L^4) / (24*EI)
     = q*L^4 / (8*EI)
     = 1000 * 1.0^4 / (8 * 1.666e6)
     = 1000 / 13.333e6
     = 7.5e-5 m
     = 0.075 mm
```

This small deflection (0.075 mm for a 1-meter beam) confirms we are in the small-deflection regime where Euler-Bernoulli theory is valid.

---

## PINN Implementation Workflow

The PINN approach replaces traditional numerical methods (finite elements, finite differences) with a neural network trained to satisfy the physics. Here is the 9-step workflow applied to this beam problem:

### Step 1: Define the Physical Domain

The beam spans from x = 0 (fixed wall) to x = L = 1.0 m. After nondimensionalization, the domain becomes s in [0, 1].

### Step 2: Identify the Governing Equation

The 4th-order ODE: EI * w''''(x) = q, which in nondimensional form becomes v''''(s) = 1.

### Step 3: Identify All Boundary Conditions

Four BCs total - two at the fixed end (displacement and slope) and two at the free end (moment and shear).

### Step 4: Choose Hard vs. Soft Constraint Strategy

- Hard constraints: v(0) = 0 and v'(0) = 0 are enforced architecturally using the s^2 trick
- Soft constraints: v''(1) = 0 and v'''(1) = 0 are enforced via penalty terms in the loss

### Step 5: Design the Network Architecture

A small fully-connected network (3 layers of 30 neurons, tanh activation) with output multiplied by s^2.

### Step 6: Formulate the Loss Function

PDE residual (v'''' - 1)^2 at collocation points plus weighted BC penalty terms for the free-end conditions.

### Step 7: Sample Collocation Points

60 uniformly-spaced points in [0, 1] where the PDE must be satisfied.

### Step 8: Train with Automatic Differentiation

Adam optimizer computes gradients through four levels of automatic differentiation (to get the 4th derivative).

### Step 9: Validate Against Exact Solution

Compare the trained network output to the known analytical solution.

### Comparison to Other PINN Projects

| Aspect | Channel Flow | Thermoelastic Bar | Cantilever Beam |
|--------|-------------|-------------------|-----------------|
| PDE Order | 2nd order | 2nd order (coupled) | 4th order |
| Number of PDEs | 1 | 2 (coupled) | 1 |
| Domain | 1D (y in [-1,1]) | 1D (x in [0,1]) | 1D (s in [0,1]) |
| BC Type | Dirichlet only | Dirichlet + Neumann | Dirichlet + Neumann |
| Hard Constraints | (1-y^2)*N(y) | Trial functions | s^2 * N(s) |
| Soft Constraints | None | Neumann BCs | Free-end BCs |
| Derivatives Needed | 2nd | 2nd | 4th |
| Physical Output | Velocity | Temperature + Displacement | Deflection |
| Nondimensionalized | Yes | Yes | Yes |
| Collocation Points | 50 | 64 | 60 |

---

## Network Architecture

```
Input: s (nondimensional position, 0 to 1)
  |
  v
[Linear: 1 -> 30] -> [tanh]
  |
  v
[Linear: 30 -> 30] -> [tanh]
  |
  v
[Linear: 30 -> 30] -> [tanh]
  |
  v
[Linear: 30 -> 1]
  |
  v
Raw network output: N(s)
  |
  v
Multiply by s^2:  v(s) = s^2 * N(s)
  |
  v
Output: nondimensional deflection v(s)
```

### Why s^2?

The factor s^2 is the key architectural choice. When s = 0:

- v(0) = 0^2 * N(0) = 0 (zero displacement - satisfied exactly)
- v'(0) = 2*0*N(0) + 0^2*N'(0) = 0 (zero slope - satisfied exactly)

This means no matter what the network N(s) outputs, the fixed-end boundary conditions are always satisfied. The network only needs to learn the interior behavior and the free-end conditions.

### Why tanh?

The tanh activation function is smooth and infinitely differentiable. Since we need the 4th derivative of the network output for the beam equation, we need an activation whose derivatives remain well-behaved through four levels of differentiation. ReLU would fail here (its higher derivatives are zero or undefined). Tanh provides smooth gradients at all orders.

### Why 3 Layers of 30 Neurons?

The beam deflection is a simple 4th-degree polynomial. A small network is sufficient to represent this - larger networks would train more slowly and risk overfitting to noise in the optimization. Three hidden layers provide enough depth for the network to learn the polynomial shape without unnecessary complexity.

---

## Hard Boundary Constraints

### The s^2 Trick for Fixed-End Conditions

At a clamped (fixed) support, both the deflection and the slope must be zero. We enforce both simultaneously through a single architectural modification:

```
v(s) = s^2 * N(s)
```

where N(s) is the raw neural network output.

### Why This Works - Mathematical Proof

**Zero displacement at s = 0:**

```
v(0) = 0^2 * N(0) = 0    [always true, regardless of N]
```

**Zero slope at s = 0:**

Using the product rule:

```
v'(s) = 2*s * N(s) + s^2 * N'(s)
v'(0) = 2*0 * N(0) + 0^2 * N'(0) = 0    [always true, regardless of N or N']
```

Both conditions hold for any network weights - they are embedded in the structure, not learned.

### Why Not Enforce All Four BCs with Hard Constraints?

In principle, one could design an ansatz that satisfies all four boundary conditions. For example, a trial function of the form:

```
v(s) = s^2 * [a*(1-s)^2 + b*(1-s) + c] * N(s)
```

could be tuned to satisfy the free-end conditions too. However, this introduces coupling between the boundary enforcement and the network's freedom to represent the interior solution. The simpler approach - hard constraints for two BCs and soft constraints for the other two - gives the network maximum flexibility while guaranteeing the most restrictive conditions (the fixed end).

### Comparison to Other Hard Constraint Approaches

| Project | Hard Constraint Form | BCs Enforced |
|---------|---------------------|--------------|
| Channel Flow | (1 - y^2) * N(y) | u(-1) = 0, u(1) = 0 |
| Cantilever Beam | s^2 * N(s) | v(0) = 0, v'(0) = 0 |

In both cases, the multiplying factor vanishes at the boundary, forcing the output to zero. For the beam, s^2 (rather than just s) ensures that the first derivative also vanishes - a more restrictive constraint matching the physics of a clamped support.

---

## Physics Residual and Loss Function

The total loss has two components: the PDE residual enforcing the beam equation everywhere, and penalty terms enforcing the free-end boundary conditions.

### Nondimensionalization

Before computing the loss, we scale the problem to order-1 quantities:

```
s = x / L              (nondimensional position, range [0, 1])
v = w / w_scale        (nondimensional deflection)
w_scale = q*L^4 / EI = 1000 * 1.0^4 / 1.666e6 = 6.0e-4 m
```

The governing equation in nondimensional form becomes:

```
v''''(s) = 1
```

This is much easier for the network to learn than the dimensional form, because both the input (s) and the target derivative (1) are of order unity.

### PDE Residual

At each collocation point s_i, we compute the 4th derivative of the network output using automatic differentiation and form the residual:

```
R_PDE = (1/N) * sum_i [ v''''(s_i) - 1 ]^2
```

This term drives the network to satisfy the beam equation at every sampled point.

### Soft Boundary Condition Penalty

The free-end conditions are enforced as penalty terms:

```
R_BC = weight * [ v''(1)^2 + v'''(1)^2 ]
```

where weight = 10 amplifies the boundary terms relative to the interior residual.

### Total Loss

```
Loss = R_PDE + R_BC
     = (1/N) * sum_i [v''''(s_i) - 1]^2  +  10 * [v''(1)^2 + v'''(1)^2]
```

### Why Weight = 10?

The PDE residual is averaged over 60 points, producing a small per-point contribution. The boundary conditions, evaluated at single points, need amplification to compete. A weight of 10 ensures the optimizer prioritizes satisfying the boundary conditions while still learning the interior physics. Too large a weight would cause the optimizer to ignore the PDE; too small and the boundaries would be poorly satisfied.

### Comparison of Loss Structures

| Project | PDE Terms | BC Penalty Terms | Weights |
|---------|-----------|-----------------|---------|
| Channel Flow | (u'' + 2)^2 | None (all hard) | N/A |
| Thermoelastic Bar | Temperature + Displacement residuals | Neumann BCs | 10 |
| Cantilever Beam | (v'''' - 1)^2 | v''(1)^2 + v'''(1)^2 | 10 |

---

## Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | Adam | Adaptive learning rates handle varying gradient magnitudes across 4 derivative levels |
| Learning Rate | 1e-3 | Standard starting rate for Adam; not too aggressive for a smooth target |
| Scheduler | StepLR | Halves learning rate at fixed intervals for stable convergence |
| Step Size | 4000 epochs | Allows sufficient exploration before reducing step size |
| Gamma | 0.5 | Halves the learning rate each step |
| Total Epochs | 8000 | Two learning rate phases: 1e-3 for epochs 0-3999, 5e-4 for epochs 4000-7999 |
| Collocation Points | 60 | Uniform spacing in [0, 1]; sufficient for a smooth polynomial solution |
| BC Weight | 10 | Balances boundary accuracy against interior PDE satisfaction |
| Best-State Tracking | Yes | Saves model weights at lowest loss; guards against late-training instability |

### Why Best-State Tracking?

During training, the loss may occasionally spike due to the optimizer overshooting in parameter space. By tracking the best (lowest-loss) model state throughout training and restoring it at the end, we guarantee that the final model is at least as good as the best intermediate state. This is especially important when training with a learning rate schedule - the model might pass through an excellent state before the rate reduction and then wander slightly after.

---

## Results

The trained PINN reproduces the exact cantilever beam deflection curve with high accuracy.

<img width="2380" height="877" alt="image" src="https://github.com/user-attachments/assets/014ea234-0bd9-48e1-9ee0-cb87ae899f87" />


### Key Observations

- The PINN output closely matches the exact analytical solution across the entire beam span
- Maximum deflection occurs at the free tip (s = 1), matching the expected value of w(L) = 0.075 mm
- The deflection curve has the characteristic parabolic-quartic shape: rising steeply near the fixed end and flattening near the free end
- Hard boundary constraints ensure perfect agreement at the fixed end (zero displacement and zero slope)
- Free-end conditions (zero moment and zero shear) are satisfied to high accuracy through the soft penalty

### Physical Interpretation of the Deflection Shape

The beam curves most sharply near the fixed end where the bending moment is largest (the wall must resist the total load), and becomes nearly straight near the free tip where the moment approaches zero. This is the characteristic shape seen in any diving board or cantilevered shelf - most of the curvature is near the support.

---

## Comparison to Other Projects

| Feature | Channel Flow | Thermoelastic Bar | Cantilever Beam |
|---------|-------------|-------------------|-----------------|
| Physics | Viscous fluid flow | Heat + thermal stress | Structural bending |
| PDE Type | Poisson (2nd order) | Coupled 2nd order | 4th order single |
| Difficulty | Simplest | Medium (coupled) | Medium (high-order) |
| AD Depth | 2 levels | 2 levels | 4 levels |
| Hard BCs | Both walls | Fixed temperature | Clamped end |
| Soft BCs | None | Stress-free end | Moment-free + Shear-free |
| Solution Shape | Parabola | Linear + Quadratic | Quartic polynomial |
| Key Challenge | None (simple) | Coupling two PDEs | 4th derivatives + mixed BC types |
| Network Size | 3x20 | 3x30 (two outputs) | 3x30 |
| Epochs | 5000 | 15000 | 8000 |

### Progressive Learning Path

These three projects form a natural progression:

1. **Channel Flow** - introduces the PINN concept with the simplest possible PDE and all-hard boundary conditions
2. **Thermoelastic Bar** - adds coupled equations and mixed hard/soft boundary enforcement
3. **Cantilever Beam** - introduces 4th-order differentiation and the challenge of computing stable higher-order derivatives through automatic differentiation

---

## Assumptions and Limitations

### Physical Assumptions

- **Small deflections**: The tip deflection (0.075 mm) is much smaller than the beam length (1000 mm), so geometric nonlinearity is negligible
- **Linear elastic material**: Steel at these stress levels behaves linearly (Hooke's law holds)
- **Euler-Bernoulli kinematics**: Cross-sections remain plane and perpendicular to the neutral axis (no shear deformation)
- **Uniform cross-section**: E and I are constant along the beam length
- **Static loading**: No time dependence, inertia, or dynamic effects
- **No axial loading**: Pure transverse bending only

### PINN Limitations

- **1D only**: This implementation handles a single spatial dimension; real beams are 3D structures
- **Known load distribution**: The load q must be specified; inverse problems (inferring q from measurements) would require a different formulation
- **Smooth solutions only**: The current network and collocation scheme assume a smooth deflection curve; point loads or discontinuities would need special treatment
- **4th derivative sensitivity**: Computing four levels of automatic differentiation amplifies numerical noise; careful architecture choices (tanh, sufficient width) are essential
- **Single load case**: The trained network solves one specific loading scenario; changing q, L, or EI requires retraining

---

## Possible Extensions

1. **Point loads and mixed loading** - Replace uniform q with concentrated forces or combinations; requires handling discontinuities in shear and moment
2. **Variable cross-section** - Let I vary along the beam (tapered beam); the ODE becomes (EI(x)*w'')'' = q
3. **Multiple spans and supports** - Continuous beams over several supports; introduces internal compatibility conditions
4. **Dynamic vibration** - Add inertia term: EI*w'''' + rho*A*w_tt = q(x,t); the PINN must learn both space and time
5. **Large deflections** - Geometric nonlinearity for flexible structures; the governing equation becomes nonlinear
6. **Inverse problems** - Given measured deflections at a few points, infer the load distribution or stiffness variation
7. **2D plate bending** - Extend to the biharmonic equation for thin plates: D*nabla^4(w) = q(x,y)
8. **Parametric PINN** - Include E, I, q, L as additional network inputs to create a surrogate model for rapid design exploration
9. **Timoshenko beam theory** - Include shear deformation for thick beams; introduces a coupled system of two 2nd-order ODEs
10. **Buckling analysis** - Add axial compression to find critical loads; the eigenvalue problem P_cr = pi^2*EI/L^2

---

## Requirements

```
torch >= 1.10
numpy
matplotlib
```

No GPU required. Training completes in under one minute on a standard CPU.

---

## How to Run

```bash
cd Cantilever_Beam_Deflection
python cantilever_beam_deflection.py
```

The script will:
1. Train the PINN for 8000 epochs, printing loss every 1000 epochs
2. Restore the best model state (lowest loss encountered during training)
3. Plot the predicted deflection against the exact analytical solution
4. Display the relative error across the beam span

---

## Code Walkthrough

For a detailed line-by-line explanation of the implementation, see [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md).
