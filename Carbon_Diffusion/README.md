# Carbon Diffusion - Physics-Informed Neural Network



<img width="1327" height="861" alt="carbon_diffusion_diagram" src="https://github.com/user-attachments/assets/ab2ec0d4-8806-4c86-807c-996a50627208" />

## Background

This project solves the first **time-dependent partial differential equation** (PDE) in the repository. All previous projects dealt with steady-state ordinary differential equations (ODEs) - problems where the solution does not change with time. Here, we tackle transient carbon diffusion into steel, a process that evolves over minutes and hours.

Carburization is an industrial heat treatment where steel parts are exposed to a carbon-rich atmosphere at high temperature. Carbon atoms migrate from the surface into the bulk, creating a hardened outer layer. The concentration of carbon decreases with depth, forming a characteristic S-shaped profile described by the complementary error function (erfc). This profile is not static - it spreads deeper into the material as time passes.

A physics-informed neural network (PINN) learns to predict the carbon concentration at any depth and any time simultaneously, satisfying both the governing PDE and the boundary/initial conditions.

---

## Key Terminology

| Term | Definition |
| --- | --- |
| Diffusion | The net movement of atoms from regions of high concentration to regions of low concentration, driven by random thermal motion |
| Fick's second law | The PDE governing how concentration changes with both position and time: C_t = D * C_xx |
| Diffusivity (D) | A material property quantifying how fast atoms spread; units of m^2/s |
| erfc | The complementary error function, erfc(z) = 1 - erf(z); it naturally appears as the solution to diffusion into a semi-infinite solid |
| Similarity variable | A combination of x and t (here eta = x / (2*sqrt(D*t))) that collapses the PDE into an ODE, revealing that profiles at different times are self-similar |
| Sobol sequence | A quasi-random, low-discrepancy sequence that fills space more uniformly than pseudorandom numbers, reducing gaps and clusters |
| Quasi-random | Deterministic sequences designed to cover a domain evenly; unlike truly random points, they avoid leaving large empty regions |
| Boundary layer | The thin region near the surface where concentration changes rapidly from the surface value to the interior value |
| Sigmoid | The function sigma(z) = 1/(1+exp(-z)), which maps any real number to the interval (0, 1) |
| Feature engineering | Adding derived input variables (like sqrt(tau)) that encode known physics, making the network's job easier |
| Initial condition (IC) | The state of the system at time zero: C(x, 0) = 0 everywhere inside the steel |
| Boundary condition (BC) | Constraints at spatial boundaries: C(0, t) = 1 at the surface, C(L, t) = 0 at the far end |

---

## Physical Problem

**The carburization process:**

A steel sample initially contains no dissolved carbon (or negligible carbon compared to the surface level). At time t = 0, the surface is exposed to a carbon-rich gas that maintains a constant carbon concentration at x = 0. Carbon atoms then diffuse inward following Fick's law.

**Physical parameters:**

| Parameter | Value | Meaning |
| --- | --- | --- |
| D | 1e-11 m^2/s | Carbon diffusivity in austenitic steel at ~930 C |
| L | 0.002 m (2 mm) | Modeled depth into the steel |
| t_final | 3600 s (1 hour) | Duration of carburization |
| C(0, t) | 1 (normalized) | Constant surface concentration |
| C(L, t) | 0 | Far-field condition (no carbon has reached this depth) |
| C(x, 0) | 0 | Initially carbon-free interior |

**Why L = 2 mm is effectively infinite:**

The characteristic diffusion penetration depth is sqrt(D * t). At t = 3600 s:

```
penetration depth = sqrt(1e-11 * 3600) = sqrt(3.6e-8) ~ 0.00019 m = 0.19 mm

```

Since 0.19 mm is much less than 2 mm, carbon never reaches the far boundary during the simulation. The domain behaves as semi-infinite, which is exactly the assumption behind the erfc analytical solution.

**Physical intuition - why the erfc shape forms:**

Imagine releasing a large number of random walkers at the surface. Each atom takes a random step left or right at each instant. Near the surface, many atoms are present, so the concentration is high. Further away, fewer atoms have wandered that far, creating a smooth decay. The erfc function captures exactly this statistical spreading. The profile is steep near x = 0 (strong gradient drives fast flux) and flattens out deeper inside (few atoms, weak gradient).

---

## Governing Equation

Fick's second law in one spatial dimension:

```
dC/dt = D * d^2C/dx^2

```

where:

- C(x, t) is the carbon concentration (normalized between 0 and 1)
- x is depth from the surface (meters)
- t is time (seconds)
- D is the diffusion coefficient (m^2/s)

This equation states that the local rate of concentration change equals the diffusivity times the curvature of the concentration profile. Where the profile is concave up (C_xx > 0), concentration increases over time. Where it is concave down, concentration decreases.

---

## Analytical Solution - Step by Step

**Step 1: Identify the problem type**

We have a linear PDE with:

- Semi-infinite domain: 0 <= x < infinity
- Constant BC at x = 0: C(0, t) = 1
- Far-field condition: C(infinity, t) = 0
- Uniform IC: C(x, 0) = 0

**Step 2: Introduce the similarity variable**

Define:

```
eta = x / (2 * sqrt(D * t))

```

This single variable combines x and t. The physical intuition is that the diffusion front position scales as sqrt(t), so distances should be measured relative to this moving scale.

**Step 3: Transform the PDE**

Let C(x, t) = f(eta). Compute the partial derivatives:

```
dC/dt = f'(eta) * d(eta)/dt = f'(eta) * (-x / (4 * D^(1/2) * t^(3/2))) = f'(eta) * (-eta / (2t))

d^2C/dx^2 = f''(eta) * (1 / (4*D*t))

```

Substituting into C_t = D * C_xx:

```
f'(eta) * (-eta / (2t)) = D * f''(eta) * (1 / (4*D*t))

```

Simplifying (multiply both sides by 4t):

```
-2 * eta * f'(eta) = f''(eta)

```

**Step 4: Solve the ODE**

The equation f'' + 2*eta*f' = 0 has the solution:

```
f(eta) = A + B * erf(eta)

```

where erf is the error function.

**Step 5: Apply boundary conditions**

- At x = 0 (eta = 0): C = 1, so f(0) = A + B * erf(0) = A = 1
- At x -> infinity (eta -> infinity): C = 0, so f(inf) = 1 + B * 1 = 0, giving B = -1

**Step 6: Write the final solution**

```
C(x, t) = 1 - erf(x / (2*sqrt(D*t))) = erfc(x / (2*sqrt(D*t)))

```

**Verification:** At x = 0, erfc(0) = 1 (surface condition satisfied). As x -> infinity, erfc(infinity) = 0 (far-field satisfied). At t = 0+ for any x > 0, eta -> infinity, so C -> 0 (initial condition satisfied).

---

## Nondimensionalization

To improve numerical conditioning, the code scales variables to O(1):

```
x* = x / L           (spatial coordinate in [0, 1])
tau = t / t_final     (time coordinate in [0, 1])

```

The PDE in nondimensional form becomes:

```
(1/t_final) * dC/d(tau) = (D/L^2) * d^2C/d(x*^2)

```

The network learns C(x*, tau) directly in the unit square [0,1] x [0,1], with the physical constants appearing in the residual calculation.

---

## PINN Implementation Workflow

```
1. Define network architecture
2. Engineer input features [x*, tau, sqrt(tau + epsilon)]
3. Design output transform to enforce constraints
4. Generate training points (Sobol + biased sampling)
5. Handle corner singularity at (0, 0)
6. Define composite loss (PDE + weighted BCs + weighted IC)
7. Train with Adam optimizer + learning rate scheduling
8. Track best model state
9. Evaluate at multiple time snapshots
10. Compare against analytical erfc solution

```

---

## Architecture

```
Input: 3 features -> [40] -> [40] -> [40] -> [40] -> 1 output
         tanh       tanh     tanh     tanh

```

| Component | Choice | Reason |
| --- | --- | --- |
| Hidden layers | 4 layers of 40 neurons | Sufficient capacity for smooth diffusion profiles without overfitting |
| Activation | tanh | Smooth, differentiable everywhere; essential for computing d^2C/dx^2 via automatic differentiation |
| Input dimension | 3 | x*, tau, and engineered feature sqrt(tau) |
| Output dimension | 1 | Scalar concentration C |

The architecture is deeper than the 2-3 layer networks used for steady-state ODEs in earlier projects because the network must represent variation across both space and time simultaneously.

---

## Feature Engineering

The network receives three inputs:

```
[x*, tau, sqrt(tau + 1e-6)]

```

**Why sqrt(tau)?**

The analytical solution shows that diffusion penetration scales as sqrt(t). The similarity variable eta = x / (2*sqrt(D*t)) means the network must internally learn to compute a ratio involving sqrt(t). By providing sqrt(tau) as a direct input, we encode this known physics into the representation.

Without this feature, the network must learn the square-root relationship from data alone - possible but slower and less accurate. With it, the network can form linear combinations of x* and sqrt(tau) in the first layer, directly approximating the similarity variable.

The small epsilon (1e-6) prevents division by zero or undefined gradients at tau = 0.

---

## Output Transform

The raw network output (a scalar) passes through:

```
C_predicted = (1 - x*) * sigmoid(net_output)

```

This enforces two things simultaneously:

1. **sigmoid(...)** maps the output to (0, 1), ensuring the concentration stays physically bounded. Carbon concentration cannot be negative or exceed the surface value.
2. *(1 - x)** forces C = 0 at x* = 1 (i.e., x = L), hard-coding the far-field boundary condition. The network cannot violate this constraint regardless of its weights.

The surface boundary condition C(0, t) = 1 is NOT hard-coded - it is enforced through the loss function. At x* = 0, the transform gives sigmoid(net_output), which the training drives toward 1.

---

## Sampling Strategy

### Sobol quasi-random sampling

Instead of uniform grids or pseudorandom points, the code uses Sobol sequences. These are low-discrepancy sequences that fill the 2D (x*, tau) domain more evenly than random sampling. The benefit: fewer points are needed to achieve good coverage, and there are no accidental gaps where the PDE goes unenforced.

### Boundary-layer biased sampling

The steepest gradients occur near x* = 0 and near tau = 0 (early times when the profile is sharpest). The code generates biased points by squaring the Sobol coordinates:

```
x_biased = x_sobol^2      (clusters points near x* = 0)
tau_biased = tau_sobol^2   (clusters points near tau = 0)

```

Squaring a uniform [0,1] variable shifts the distribution toward 0 - more points land where accuracy matters most.

### Point budget

| Point type | Count | Purpose |
| --- | --- | --- |
| Interior (uniform Sobol) | 500 | PDE residual enforcement across full domain |
| Interior (biased) | 500 | Extra PDE accuracy near surface and early times |
| Surface BC (x* = 0) | 250 | Enforce C(0, tau) = 1 |
| Initial condition (tau = 0) | 250 | Enforce C(x*, 0) = 0 |
| **Total** | **1500** |  |

### Corner singularity exclusion

At the point (x*, tau) = (0, 0), the boundary condition says C = 1 but the initial condition says C = 0. This is a mathematical discontinuity - the exact solution has a jump at this corner. Including training points here would force the network to satisfy contradictory constraints, destabilizing training.

The code excludes points near this corner, letting the network interpolate smoothly between the two conditions.

---

## Loss Function

```
Loss = loss_pde + 20 * loss_surface + 20 * loss_initial

```

| Component | Formula | Weight | Role |
| --- | --- | --- | --- |
| loss_pde | MSE of (C_t - D*C_xx) at interior points | 1 | Enforce the diffusion equation |
| loss_surface | MSE of (C(0, tau) - 1) at surface points | 20 | Enforce constant surface concentration |
| loss_initial | MSE of (C(x*, 0) - 0) at IC points | 20 | Enforce zero initial concentration |

The boundary/initial conditions are weighted 20x higher than the PDE residual. Without this weighting, the network might minimize PDE error while drifting away from the correct boundary values. The conditions anchor the solution, and the PDE propagates that information inward.

---

## Training Configuration

| Setting | Value | Rationale |
| --- | --- | --- |
| Optimizer | Adam | Adaptive learning rates handle the multi-scale loss landscape |
| Initial learning rate | 1e-3 (default Adam) | Standard starting point |
| Scheduler | StepLR, halve every 5000 epochs | Coarse-to-fine optimization; large steps early, refined later |
| Total epochs | 12000 | Sufficient for convergence on this problem |
| Best-state tracking | Yes | Saves the model weights with lowest total loss, protecting against late-training instability |

---

## Results



<img width="2578" height="977" alt="image" src="https://github.com/user-attachments/assets/56a05b84-322b-4673-ac29-2a54c679c9fc" />


The output figure contains two panels:

**Left panel - Concentration profiles:**

- Three curves showing C(x) at t = 900 s, 1800 s, and 3600 s
- PINN predictions (solid or markers) overlaid on analytical erfc solutions (dashed)
- The profiles spread deeper into the steel with increasing time
- At t = 900 s, carbon has barely penetrated 0.2 mm
- At t = 3600 s, appreciable concentration extends to about 0.4 mm
- Agreement between PINN and analytical solution validates the approach

**Right panel - Training convergence:**

- Loss vs. epoch on logarithmic scale
- Shows the characteristic rapid initial descent followed by slower refinement
- Learning rate drops are visible as changes in descent rate at epoch 5000 and 10000

---

## Comparison to Other Projects in the Repository

| Project | Equation Type | Inputs | Key Technique | Difficulty |
| --- | --- | --- | --- | --- |
| Non_Unique_BVP | Steady ODE | 1 (x) | Multiple solutions from different ICs | Low |
| Autograd_Verification | Steady ODE | 1 (x) | Gradient validation | Low |
| Architecture_Comparison | Steady ODE | 1 (x) | Width/depth/activation sweep | Low-Medium |
| Heat_Conduction_With_Source | Steady ODE | 1 (x) | Source term, exact solution comparison | Medium |
| Cantilever_Beam_Deflection | 4th-order ODE | 1 (x) | Higher-order derivatives, beam physics | Medium |
| Pressure_Driven_Channel_Flow | Steady ODE (2nd order) | 1 (y) | Fluid mechanics, Poiseuille flow | Medium |
| Coupled_Thermoelastic_Bar | Coupled steady ODEs | 1 (x) | System of equations, multi-output network | Medium-High |
| Inverse_Heat_Conduction | Steady ODE | 1 (x) | Unknown parameter estimation | Medium-High |
| **Carbon_Diffusion** | **Transient PDE** | *3 (x, tau, sqrt(tau))** | **Time dependence, feature engineering, Sobol sampling, biased points, sigmoid bounding, corner handling** | **High** |

**What makes this project uniquely challenging:**

1. **Partial vs. ordinary:** The network must learn a function of two independent variables (space and time), not just one. The solution surface lives in 3D, not on a curve.
2. **Mixed derivatives:** Computing the PDE residual requires both dC/dt and d^2C/dx^2 from the same network - two different derivative directions.
3. **Singular corner:** The incompatibility between IC and BC at (0,0) has no analog in steady-state problems and requires special handling.
4. **Sharp gradients that move:** The diffusion front is thin and migrates with time. Uniform sampling misses it; biased sampling is essential.
5. **Multiple time scales:** Early-time profiles are extremely steep; late-time profiles are gentle. The network must represent both regimes.

---

## Assumptions and Limitations

1. **Constant diffusivity** - In reality, D depends on temperature and sometimes on concentration. Here D is fixed at 1e-11 m^2/s.
2. **One-dimensional** - Diffusion is treated as occurring in one direction only (depth). This is valid for flat surfaces or when the radius of curvature is much larger than the diffusion depth.
3. **Constant surface concentration** - The gas-phase carbon activity is assumed constant. In practice, furnace conditions may fluctuate.
4. **No phase transformations** - At high carbon concentrations, carbide phases can form, altering the diffusion behavior. This model assumes a single-phase solid solution.
5. **Semi-infinite approximation** - The 2 mm domain is treated as infinite. For longer times or higher diffusivities, the far-boundary condition would need revision.
6. **Normalized concentration** - The model works in dimensionless C in [0, 1]. Mapping back to weight percent requires knowing the actual surface concentration.

---

## Extensions

1. **Concentration-dependent diffusivity** - Replace constant D with D(C), making the PDE nonlinear. The PINN framework handles this naturally since the residual is evaluated pointwise.
2. **Two and three dimensions** - Add y and z inputs for diffusion around corners, notches, or cylindrical geometries. The network architecture scales by adding input neurons.
3. **Time-varying boundary conditions** - Model a carburization cycle with temperature ramps (and thus changing D and surface C). The PINN can accept time-dependent BCs through the loss function.
4. **Multi-species diffusion** - Extend to coupled carbon-nitrogen diffusion (carbonitriding) by adding output neurons and coupled PDE residuals, similar to the Coupled_Thermoelastic_Bar approach.
5. **Inverse problem** - Given measured concentration profiles, infer the unknown diffusivity D. This combines techniques from this project with those from Inverse_Heat_Conduction.
6. **Adaptive sampling** - Dynamically add collocation points where the PDE residual is largest during training, concentrating computational effort where it is most needed.
7. **Transfer learning** - Pre-train on a simple diffusion case, then fine-tune for more complex geometries or nonlinear diffusivities. The learned feature representations (especially the sqrt(t) scaling) transfer well.

---

## File Structure

```
Carbon_Diffusion/
    carbon_diffusion.py       # Main PINN training and evaluation script
    chapter7_diffusion.png    # Output: concentration profiles and convergence plot
    README.md                 # This file

```

---

## Running the Code

```bash
cd Carbon_Diffusion
python carbon_diffusion.py

```

The script trains for 12000 epochs (typically 3-8 minutes on a modern GPU, longer on CPU) and saves the results figure automatically.

---

## References

- Fick, A. (1855). On liquid diffusion. Philosophical Magazine, 10, 30-39.
- Carslaw, H.S. and Jaeger, J.C. (1959). Conduction of Heat in Solids. Oxford University Press.
- Raissi, M., Perdikaris, P., and Karniadakis, G.E. (2019). Physics-informed neural networks. Journal of Computational Physics, 378, 686-707.

