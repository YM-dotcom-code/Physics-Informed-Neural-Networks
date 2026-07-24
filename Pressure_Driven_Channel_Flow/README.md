# Physics-Informed Neural Network: Pressure-Driven Channel Flow

A PINN that solves the Stokes (Poiseuille) flow problem — steady, pressure-driven viscous flow between two infinite parallel plates — using **no training data**, only the governing equation.

---

## Table of Contents

1. [Background](#background)
2. [Physical Problem](#physical-problem)
3. [Governing Equation](#governing-equation)
4. [Analytical Solution (Step by Step)](#analytical-solution-step-by-step)
5. [PINN Implementation](#pinn-implementation)
6. [Hard Boundary Constraints](#hard-boundary-constraints)
7. [Residual Normalization](#residual-normalization)
8. [Training Configuration](#training-configuration)
9. [Results](#results)
10. [How to Run](#how-to-run)
11. [CODE_WALKTHROUGH.md](#code_walkthroughmd)

---

## Background

### What Is Poiseuille Flow?

Poiseuille (or Stokes) flow is one of the simplest solutions in fluid mechanics. It describes the velocity profile of a viscous fluid flowing between two stationary parallel plates, driven by a constant pressure gradient. The result is the classic **parabolic velocity profile** — fastest in the middle, zero at the walls.

### Why Solve This With a PINN?

This problem has an exact analytical solution, which makes it a perfect **benchmark** for testing whether a PINN can:
- Learn a smooth parabolic profile from a single ODE
- Enforce no-slip boundary conditions via hard constraints
- Converge to machine-precision accuracy

It's a stepping stone between simple 1D problems (like the thermoelastic bar) and more complex fluid dynamics (Navier-Stokes).

---

## Physical Problem

```
        Stationary wall (y = +h)
    ════════════════════════════════════
    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   u = 0 (no-slip)
    
           ──→  ──→  ───→  ──→  ──→        Flow direction
                     ────→                  (max at center)
           ──→  ──→  ───→  ──→  ──→
    
    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   u = 0 (no-slip)
    ════════════════════════════════════
        Stationary wall (y = −h)
    
    ←────── Pressure gradient dp/dx ──────→
```

### Setup
- **Geometry**: Two infinite, parallel, stationary plates separated by distance 2h
- **Fluid**: Incompressible, Newtonian (viscosity μ)
- **Driving force**: Constant pressure gradient dp/dx in the flow direction
- **Domain**: y ∈ [−h, +h] (across the channel)

### Parameters Used

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Dynamic viscosity | μ | 1.0 | Pa·s |
| Pressure gradient | dp/dx | −2.0 | Pa/m |
| Half-channel height | h | 1.0 | m |

---

## Governing Equation

The Navier-Stokes equations for steady, fully-developed, incompressible flow between parallel plates reduce to a single ODE:

```
μ · u″(y) = dp/dx
```

Where:
- `u(y)` = velocity in the flow direction (what we want to find)
- `u″(y)` = second derivative of velocity with respect to y
- `μ` = dynamic viscosity (resistance to flow)
- `dp/dx` = pressure gradient (the driving force)

### Boundary Conditions

| Location | Condition | Physical Meaning |
|----------|-----------|-----------------|
| y = −h | u(−h) = 0 | No-slip at bottom wall |
| y = +h | u(+h) = 0 | No-slip at top wall |

**No-slip** means the fluid velocity is zero at a solid wall — the fluid "sticks" to the surface.

---

## Analytical Solution (Step by Step)

Since we have an exact solution, we can verify whether the PINN learns the correct answer.

### Step 1: Start with the ODE

```
μ · u″(y) = dp/dx
```

Divide both sides by μ:

```
u″(y) = dp/dx / μ = −2 / 1 = −2
```

### Step 2: Integrate once

```
u′(y) = −2y + C₁
```

### Step 3: Integrate again

```
u(y) = −y² + C₁y + C₂
```

### Step 4: Apply boundary condition u(−h) = u(−1) = 0

```
0 = −(−1)² + C₁(−1) + C₂
0 = −1 − C₁ + C₂
→  C₂ = 1 + C₁     ... (i)
```

### Step 5: Apply boundary condition u(+h) = u(+1) = 0

```
0 = −(1)² + C₁(1) + C₂
0 = −1 + C₁ + C₂
→  C₂ = 1 − C₁     ... (ii)
```

### Step 6: Solve for constants

From (i) and (ii): `1 + C₁ = 1 − C₁` → `2C₁ = 0` → **C₁ = 0**

Substituting back: **C₂ = 1**

### Final Solution

```
u(y) = 1 − y²
```

Or in general form:

```
u(y) = −(dp/dx) · (h² − y²) / (2μ)
```

### Verification

| y position | u(y) = 1 − y² | Check |
|-----------|---------------|-------|
| y = −1 (bottom wall) | 1 − 1 = 0 | ✓ No-slip satisfied |
| y = −0.5 | 1 − 0.25 = 0.75 m/s | |
| y = 0 (centerline) | 1 − 0 = **1.0 m/s** | Maximum velocity |
| y = +0.5 | 1 − 0.25 = 0.75 m/s | |
| y = +1 (top wall) | 1 − 1 = 0 | ✓ No-slip satisfied |

### Check the PDE

```
u(y) = 1 − y²
u′(y) = −2y
u″(y) = −2

μ · u″ = 1.0 × (−2) = −2 = dp/dx  ✓
```

---

## PINN Implementation

### Network Architecture

```
Input: y (position)
    │
    ▼ normalize: s = y / h
    │
    ├── Linear(1 → 30) + Tanh
    ├── Linear(30 → 30) + Tanh
    ├── Linear(30 → 30) + Tanh
    └── Linear(30 → 1)  →  raw_velocity
    │
    ▼ hard constraint: u = velocity_scale × (1 − s²) × raw_velocity
    │
Output: u(y) (velocity)
```

- **Input**: Single coordinate y (position across channel)
- **Hidden layers**: 3 layers × 30 neurons each
- **Activation**: Tanh (smooth, infinitely differentiable — required for computing u″)
- **Output**: Single value — velocity at that position
- **Input normalization**: s = y/h maps domain to [−1, +1]

### Why This Architecture Is Simpler Than the Thermoelastic Project

| Feature | Thermoelastic Bar | Channel Flow |
|---------|-------------------|--------------|
| Equations | 2 coupled PDEs | 1 single ODE |
| Network outputs | 2 (T, u) | 1 (velocity) |
| Hidden layers | 4 × 64 | 3 × 30 |
| Loss terms | 3 (heat + mech + BC) | 1 (PDE residual only) |
| Soft BCs needed | 1 (stress) | 0 (all hard!) |

---

## Hard Boundary Constraints

Both boundary conditions are enforced **exactly** by construction:

```python
u = velocity_scale × (1 − s²) × raw_velocity
```

**How it works:**
- At the bottom wall: s = y/h = −1/1 = −1 → (1 − (−1)²) = (1−1) = **0** → u = 0 ✓
- At the top wall: s = y/h = +1/1 = +1 → (1 − (1)²) = (1−1) = **0** → u = 0 ✓
- Everywhere else: (1 − s²) > 0, so the network can output any velocity

The network **cannot violate the boundary conditions** regardless of what `raw_velocity` outputs. This means:
- Zero boundary loss terms needed
- The loss function is purely the PDE residual
- Training effort focuses entirely on satisfying the physics equation

### Why (1 − s²) Is the Perfect Choice

The exact solution is `u(y) = 1 − y² = 1 − s²` (since h = 1). This means:
- The hard constraint `(1 − s²) × raw` already has the **exact shape** of the solution
- The network only needs to learn `raw_velocity ≈ 1/velocity_scale = 1.0` (a constant!)
- This makes convergence extremely fast

---

## Residual Normalization

The PDE residual is normalized by a characteristic scale:

```python
residual = u″ − dp/dx / μ          # raw residual (should be 0)
loss = mean( (residual / residual_scale)² )   # normalized
```

Where:
```
residual_scale = |dp/dx / μ| = |−2 / 1| = 2.0
```

**Why normalize?** Without normalization, the loss magnitude depends on the physical units. Dividing by the characteristic scale makes the loss O(1), which keeps gradient magnitudes in a good range for the Adam optimizer.

---

## Training Configuration

| Setting | Value | Why |
|---------|-------|-----|
| Collocation points | 80 (uniform) | More than enough for a smooth parabola |
| Epochs | 10,000 | Converges well before this |
| Optimizer | Adam (lr = 1e-3) | Standard, adaptive learning rates |
| Scheduler | StepLR (halve every 5000) | Big steps early, fine-tune later |
| Best-state tracking | Yes | Avoids late-training noise |
| Loss function | Mean squared normalized residual | Single term — no balancing needed |

### Training Flow

```
For each of 10,000 epochs:
    1. Forward pass: compute u(y) at 80 points
    2. Autograd: compute u′(y) and u″(y)
    3. Residual: r = u″ − (dp/dx)/μ
    4. Loss: mean( (r / residual_scale)² )
    5. Backward pass: compute gradients
    6. Adam step: update weights
    7. If loss < best → save model state
    8. StepLR: check if time to halve lr

After training:
    Load best model state → evaluate on 200 points
```

---

## Results

| Metric | Value |
|--------|-------|
| Maximum error | ~10⁻⁵ to 10⁻⁶ |
| Relative L₂ error | ~10⁻⁵ to 10⁻⁶ |
| Wall error | ~10⁻¹⁶ (machine zero) |
| Centerline velocity | ~1.000000 m/s |
| Exact centerline | 1.000000 m/s |

**Key observations:**
- Wall error is machine zero because boundaries are **hard-coded** — not learned
- The network achieves ~6 digits of accuracy with a simple 3×30 architecture
- Convergence is fast because the hard constraint already has the correct shape
- No boundary loss balancing needed — single loss term makes training trivial

---

## How to Run

### Requirements
```
torch >= 2.0
numpy
matplotlib
```

### Run
```bash
python stokes_flow_pinn.py
```

### Output
- Console: training progress every 2000 epochs + final metrics
- File: `chapter4_stokes_flow.png` (velocity profile + convergence plot)

---

## CODE_WALKTHROUGH.md

For a detailed line-by-line explanation of the implementation, see [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md).
