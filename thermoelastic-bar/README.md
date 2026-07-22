
# **Coupled Thermoelastic Physics-Informed Neural Network**

This project implements a multi-output **Physics-Informed Neural Network (PINN)** for a steady one-dimensional thermoelastic problem.

A single neural network predicts two physical fields in a heated bar:

- Temperature `T(x)`
- Thermally induced axial displacement `u(x)`

The model learns these fields by enforcing the governing heat-conduction and mechanical-equilibrium equations using PyTorch automatic differentiation. No training data is needed. The governing equations themselves supervise the learning.

<p align="center">
  <img width="958" alt="Corrected thermoelastic PINN results" src="https://github.com/user-attachments/assets/56669cae-67ec-480f-a648-cf675cf880dd" />
</p>
## **Table of Contents**

- [Background](#background)
- [Physical Problem](#physical-problem)
- [Governing Equations](#governing-equations)
- [Exact Analytical Solutions](#exact-analytical-solutions)
- [PINN Architecture](#pinn-architecture)
- [Hard Boundary Constraints](#hard-boundary-constraints)
- [Physics Residuals and Loss Function](#physics-residuals-and-loss-function)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [Assumptions and Limitations](#assumptions-and-limitations)
- [Possible Extensions](#possible-extensions)
- [Requirements](#requirements)
- [Run the Project](#run-the-project)

## **Background**

### What is a Physics-Informed Neural Network

Traditional numerical methods such as the Finite Element Method (FEM) or Finite Difference Method (FDM) solve differential equations by discretizing the spatial domain into a mesh and assembling systems of algebraic equations. While effective, these methods require mesh generation, can be computationally expensive for coupled multiphysics problems, and need to be re-run for each new set of parameters.

Physics-Informed Neural Networks offer an alternative approach. They approximate the solution using a neural network and enforce the physics through the loss function. The network learns a continuous, differentiable representation of the solution field that satisfies the governing equations at sampled collocation points throughout the domain.

### What is Coupled Thermoelasticity

Thermoelasticity is the branch of continuum mechanics that studies the interaction between thermal and mechanical fields in solid bodies. When a material is heated, it expands. If the expansion is constrained or non-uniform, internal stresses develop.

In a coupled thermoelastic problem, two physical phenomena interact:

- Heat conduction determines the temperature distribution within the body.
- Thermal expansion converts the temperature field into mechanical strains and displacements.

The coupling in this project is one-way: the temperature field influences the displacement through thermal strain, but the mechanical deformation does not feed back into the temperature equation. This is valid when deformations are small and the mechanical work does not produce significant heat.

This type of problem appears in engineering applications including heated pipelines, electronic components under thermal loading, bimetallic strips, and thermally stressed structural members.

```text
┌─────────────────────┐         ┌─────────────────────┐         ┌─────────────────────┐
│                     │         │                     │         │                     │
│   Heat Conduction   │────────>│   Thermal Strain    │────────>│   Displacement      │
│                     │         │                     │         │                     │
│   k·T''(x) = 0     │         │  ε_th = α(T - Tref) │         │  u'' - α·T' = 0     │
│                     │         │                     │         │                     │
└─────────────────────┘         └─────────────────────┘         └─────────────────────┘

                              One-Way Coupling (T → u)
```

## **Physical Problem**

Consider a bar of length `L` subjected to different prescribed temperatures at its two ends. The left end is mechanically fixed, while the right end is stress-free. Heating causes the bar to expand, so the temperature field drives the mechanical displacement.

```math
T(x) \longrightarrow u(x)
```

```text
    Fixed End                                          Free End (stress-free)
    (u = 0)                                            (σ = 0)

    ████║═══════════════════════════════════════════════════════║
    ████║                                                       ║→ expansion
    ████║              Heated Bar (length L = 1 m)              ║
    ████║                                                       ║→ expansion
    ████║═══════════════════════════════════════════════════════║

    T = T₀ = 100°C                                    T = Tₗ = 500°C
    x = 0                                             x = L
```

### Problem Parameters

| Parameter | Description | Value |
|---|---|---:|
| `k` | Thermal conductivity | 50 W/(m·K) |
| `E` | Young's modulus | 200 GPa |
| `α` | Thermal expansion coefficient | 12 × 10⁻⁶ K⁻¹ |
| `L` | Bar length | 1 m |
| `T₀` | Left-end temperature | 100°C |
| `Tᴸ` | Right-end temperature | 500°C |
| `Tref` | Stress-free reference temperature | 100°C |

## **Governing Equations**

### Steady-State Heat Conduction

For steady one-dimensional heat conduction without internal heat generation:

```math
k\frac{d^2T}{dx^2}=0
```

With prescribed temperature boundary conditions:

```math
T(0)=T_0,
\qquad
T(L)=T_L
```

### Thermoelastic Constitutive Relation

The total axial strain is:

```math
\varepsilon(x)=\frac{du}{dx}
```

The thermal strain is:

```math
\varepsilon_{\mathrm{th}}(x)
=
\alpha\left[T(x)-T_{\mathrm{ref}}\right]
```

The axial stress is calculated from the difference between total strain and thermal strain:

```math
\sigma(x)
=
E\left[
\frac{du}{dx}
-
\alpha\left(T-T_{\mathrm{ref}}\right)
\right]
```

### Mechanical Equilibrium

In the absence of an axial body force, mechanical equilibrium requires:

```math
\frac{d\sigma}{dx}=0
```

Substituting the thermoelastic stress relation gives:

```math
E\left(
\frac{d^2u}{dx^2}
-
\alpha\frac{dT}{dx}
\right)=0
```

Because Young's modulus is nonzero, the displacement equation becomes:

```math
\frac{d^2u}{dx^2}
-
\alpha\frac{dT}{dx}
=0
```

This equation couples the temperature gradient to the mechanical displacement.

### Boundary Conditions

The left end of the bar is fixed:

```math
u(0)=0
```

The right end is stress-free:

```math
\sigma(L)=0
```

Using the thermoelastic constitutive equation, the stress-free condition becomes:

```math
\frac{du}{dx}(L)
=
\alpha\left[T(L)-T_{\mathrm{ref}}\right]
```

## **Exact Analytical Solutions**

### Temperature

Because the second derivative of temperature is zero, the exact temperature distribution is linear:

```math
T(x)=T_0+\frac{T_L-T_0}{L}x
```

For the parameters used in this project:

```math
T(x)=100+400x
```

```text
    T [°C]
     500 ┤                                          ●
         │                                       ╱
         │                                    ╱
         │                                 ╱
     300 ┤                              ╱
         │                           ╱
         │                        ╱
         │                     ╱
     100 ┤● ─ ─ ─ ─ ─ ─ ─ ╱
         │
         └────────────────────────────────────────── x [m]
         0                                          1.0

                    T(x) = 100 + 400x
```

### Displacement

Because mechanical equilibrium makes the axial stress constant and the free end has zero stress, the stress is zero throughout the bar:

```math
\sigma(x)=0
```

Therefore:

```math
\frac{du}{dx}
=
\alpha\left[T(x)-T_{\mathrm{ref}}\right]
```

Substituting the linear temperature field and integrating gives:

```math
u(x)
=
\alpha(T_0-T_{\mathrm{ref}})x
+
\frac{\alpha(T_L-T_0)}{2L}x^2
```

Since the reference temperature equals the left-end temperature, the first term vanishes:

```math
u(x)
=
\frac{\alpha(T_L-T_0)}{2L}x^2
```

The maximum displacement occurs at the free end:

```math
u(L)
=
\frac{\alpha(T_L-T_0)L}{2}
=
\frac{(12\times10^{-6})(500-100)(1)}{2}
=
0.0024\ \mathrm{m}
=
2400\ \mu\mathrm{m}
```

```text
    u [μm]
    2400 ┤                                          ●
         │                                       ╱
         │                                    ╱
         │                                ╱
    1200 ┤                           ╱╱
         │                      ╱╱
         │                 ╱╱
         │           ╱╱╱
         │     ╱╱╱╱
       0 ┤●╱╱
         └────────────────────────────────────────── x [m]
         0                                          1.0

              u(x) = α(Tₗ - T₀)x² / (2L)
```

## **PINN Architecture**

The neural network receives the spatial coordinate `x` and returns two outputs:

```math
x
\longrightarrow
\left[
\widehat{T}(x),
\widehat{u}(x)
\right]
```

```text
    Input          Shared Hidden Layers                    Output Heads
    Layer          (tanh activation)

                   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐
              ┌───>│ 40     │──>│ 40     │──>│ 40     │──>│ 40     │──┐
              │    │neurons │   │neurons │   │neurons │   │neurons │  │   ┌──────────┐
              │    └────────┘   └────────┘   └────────┘   └────────┘  ├──>│ T(x)     │
   ┌──────┐   │                                                       │   └──────────┘
   │ x    │───┤                                                       │
   │      │   │                                                       │   ┌──────────┐
   └──────┘   │                                                       ├──>│ u(x)     │
              │                                                       │   └──────────┘
              └───────────────────────────────────────────────────────┘
```

The architecture contains:

- One spatial input
- Four shared hidden layers with 40 neurons each
- Hyperbolic tangent activation functions
- Two separate output heads (temperature and displacement)

The shared layers learn spatial features common to both physical fields.

## **Hard Boundary Constraints**

Important boundary conditions are imposed directly through transformations of the neural-network outputs rather than relying solely on penalty terms.

The normalized spatial coordinate is:

```math
s=\frac{x}{L}
```

The temperature difference and displacement scale are:

```math
\Delta T=T_L-T_0,
\qquad
u_{\mathrm{scale}}=\alpha\Delta T L
```

The transformed temperature prediction:

```math
\widehat{T}(x)
=
T_0
+
\Delta T
\left[
s+s(1-s)N_T(s)
\right]
```

The transformed displacement prediction:

```math
\widehat{u}(x)
=
u_{\mathrm{scale}}\,sN_u(s)
```

Here, `N_T(s)` and `N_u(s)` are the raw network outputs. These transformations guarantee:

```text
    Raw Network Output                    Transformed Output (satisfies BCs exactly)

    N_T(s) ──────────────────>  T̂(x) = T₀ + ΔT·[s + s(1-s)·N_T(s)]
                                         │
                                         ├── at x=0: T̂ = T₀       ✓
                                         └── at x=L: T̂ = Tₗ       ✓

    N_u(s) ──────────────────>  û(x) = u_scale · s · N_u(s)
                                         │
                                         └── at x=0: û = 0         ✓
```

Hard constraints improve training because the network does not need to learn these conditions from penalty terms alone.

## **Physics Residuals and Loss Function**

### Residuals

PyTorch automatic differentiation calculates the required derivatives. The normalized heat-equation residual is:

```math
r_T(x)
=
\frac{
\dfrac{d^2\widehat{T}}{dx^2}
}{
\Delta T/L^2
}
```

The normalized mechanical-equilibrium residual is:

```math
r_u(x)
=
\frac{
\dfrac{d^2\widehat{u}}{dx^2}
-
\alpha\dfrac{d\widehat{T}}{dx}
}{
u_{\mathrm{scale}}/L^2
}
```

Both residuals approach zero when the PINN satisfies the governing equations.

### Loss Function

The total loss combines three components:

```math
\mathcal{L}
=
\underbrace{\frac{1}{N}\sum_{i=1}^{N}r_T(x_i)^2}_{\text{heat equation}}
+
\underbrace{\frac{1}{N}\sum_{i=1}^{N}r_u(x_i)^2}_{\text{mechanical equilibrium}}
+
100\underbrace{\left[\frac{\widehat{u}'(L)-\alpha(\widehat{T}(L)-T_{\mathrm{ref}})}{u_{\mathrm{scale}}/L}\right]^2}_{\text{stress-free BC}}
```

The boundary condition loss is weighted by 100 to prioritize satisfying the boundary conditions. All terms are nondimensionalized so that temperature and displacement errors contribute at comparable numerical scales.

## **Training Configuration**

| Setting | Value |
|---|---|
| Collocation points | 100 uniformly spaced in [0, L] |
| Optimizer | Adam |
| Initial learning rate | 1 × 10⁻³ |
| Learning rate schedule | Step decay, factor 0.5 every 5,000 epochs |
| Total epochs | 15,000 |
| Random seed | 42 |
| Device | CUDA if available, otherwise CPU |

```text
┌──────────────────────────────────────────────────────────────────────┐
│                         Training Loop                                 │
│                                                                       │
│   ┌─────────┐     ┌──────────────┐     ┌──────────────────┐         │
│   │ Sample  │────>│ Forward Pass │────>│ Compute Residuals │         │
│   │ x_i     │     │ T̂(x), û(x)  │     │ r_T, r_u          │         │
│   └─────────┘     └──────────────┘     └────────┬─────────┘         │
│                                                  │                    │
│                                                  v                    │
│   ┌─────────┐     ┌──────────────┐     ┌──────────────────┐         │
│   │ Update  │<────│ Backpropagate│<────│ Total Loss        │         │
│   │ Weights │     │ Gradients    │     │ L_heat + L_mech   │         │
│   └─────────┘     └──────────────┘     │ + 100·L_BC        │         │
│                                         └──────────────────┘         │
│                                                                       │
│   Repeat for 15,000 epochs                                           │
└──────────────────────────────────────────────────────────────────────┘
```

## **Results**

| Quantity | Result |
|---|---:|
| Final training loss | 1.53 × 10⁻⁶ |
| Maximum temperature error | 0.001°C |
| Maximum displacement error | 0.195 μm |
| Exact free-end displacement | 2400.0 μm |
| PINN free-end displacement | 2399.8 μm |

The predicted temperature and displacement curves visually overlap the corresponding analytical solutions. The loss curve contains temporary optimizer spikes at the learning rate step boundaries, but the model rapidly recovers and converges to an accurate final solution.

## **Assumptions and Limitations**

- Steady-state only (no time dependence)
- One-dimensional spatial domain
- Linear elastic material behavior
- Constant material properties (no temperature dependence)
- One-way coupling (temperature affects displacement, not the reverse)
- Small deformations (geometric nonlinearity neglected)
- No internal heat generation or body forces

## **Possible Extensions**

- Transient heat conduction with time as an additional network input
- Two-dimensional or axisymmetric thermoelastic domains
- Temperature-dependent material properties
- Two-way coupling where mechanical dissipation feeds back into the energy equation
- Inverse problems: inferring material parameters from displacement measurements
- Adaptive collocation point sampling based on residual magnitude
- Transfer learning to accelerate solutions for different boundary conditions

## **Requirements**

```bash
pip install torch numpy matplotlib
```

## **Run the Project**

```bash
python pinn_coupled_thermoelastic_bar.py
```

