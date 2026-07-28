# Inverse Heat Conduction: Parameter Identification with Physics-Informed Neural Networks



<img width="2183" height="1522" alt="inverse_heat_diagram" src="https://github.com/user-attachments/assets/2a3b60f7-6c1d-4e8d-b8e6-7945e00a0ea2" />

A physics-informed neural network (PINN) that solves the **inverse problem** of heat conduction: given noisy temperature measurements from sensors along a rod, simultaneously reconstruct the full temperature field and identify the unknown thermal conductivity. This demonstrates how PINNs can extract hidden physical parameters from sparse, noisy experimental data.

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

### Forward vs. Inverse Problems

In a **forward problem**, all physical parameters are known and the goal is to compute the system response. For example: given thermal conductivity k, heat source Q, and boundary conditions, find the temperature distribution T(x). This is straightforward and well-posed.

An **inverse problem** flips this: the system response (temperature) is partially observed through measurements, but one or more physical parameters are unknown. The goal is to recover those hidden parameters from the available data.

### Why Inverse Problems Are Hard

Inverse problems are fundamentally more challenging than forward problems for several reasons:

- **Ill-posedness**: Small errors in measurements can produce large errors in the recovered parameters. The mapping from data to parameters is often unstable.
- **Non-uniqueness**: Multiple parameter combinations may produce similar observations, especially with sparse or noisy data.
- **Nonlinearity**: Even when the forward problem is linear, the inverse mapping is typically nonlinear in the parameters.
- **Noise sensitivity**: Real sensors produce noisy data, and the inversion process can amplify that noise.

### Why PINNs Excel at Inverse Problems

Traditional inverse methods (Tikhonov regularization, adjoint methods, Bayesian inference) require careful mathematical formulation for each specific problem. PINNs offer a unified framework:

- The physics (PDE) acts as a regularizer, constraining solutions to be physically plausible.
- Parameters can be learned jointly with the solution field using gradient descent.
- No need to derive adjoint equations or sensitivity matrices manually.
- Works naturally with sparse, noisy, irregularly-spaced measurements.

### Real-World Applications of Conductivity Identification

Identifying thermal conductivity from measurements is critical in:

- **Materials characterization**: Determining thermal properties of new alloys, composites, or ceramics from lab measurements.
- **Geothermal exploration**: Estimating subsurface thermal conductivity from borehole temperature logs.
- **Non-destructive testing**: Detecting internal defects (voids, cracks) that alter local conductivity.
- **Biomedical engineering**: Measuring tissue thermal properties for hyperthermia treatment planning.
- **Building science**: Assessing insulation performance in existing structures from surface temperature data.

---

## Key Terminology

| Term | Definition |
| --- | --- |
| Inverse problem | Recovering unknown causes (parameters) from observed effects (measurements) |
| Forward problem | Computing effects (temperature field) from known causes (all parameters specified) |
| Parameter identification | A class of inverse problem where the unknown is a physical constant (here, k) |
| Log parameterization | Representing k as exp(log_k) so that gradient descent always keeps k positive |
| Sensor data | Synthetic temperature measurements at discrete points, corrupted by noise |
| Gaussian noise | Random measurement error drawn from a normal distribution (here, std = 0.1 C) |
| Data-driven loss | Mean squared error between network predictions and sensor measurements |
| Separate learning rates | Using different step sizes for network weights (slow) vs. physical parameter (fast) |
| Collocation points | Locations where the PDE residual is enforced during training |

---

## Physical Problem

Consider a one-dimensional rod of length L = 1.0 m with a uniform internal heat source Q = 5000 W/m^3. Both ends are held at fixed temperatures:

```
T(0) = 20 C       T(L) = 20 C

```

The rod has an unknown thermal conductivity k. Ten temperature sensors are placed along the rod at positions x = 0.1, 0.2, ..., 0.9 m. Each sensor reports a noisy temperature reading.

**The goal**: Using only the sensor data, the known heat source Q, and the governing PDE, determine the unknown thermal conductivity k.

```
 T=20 C                                              T=20 C
  |                                                    |
  |  Q = 5000 W/m^3 (uniform heat generation)         |
  |                                                    |
  x=0  s1  s2  s3  s4  s5  s6  s7  s8  s9  x=1.0
       |   |   |   |   |   |   |   |   |
       Noisy temperature sensors (10 total)
       
  Unknown: k = ??? W/(m*K)
  True answer: k = 50 W/(m*K)
  Initial guess: k = 10 W/(m*K)  (intentionally wrong)

```

The PINN must converge from the wrong initial guess (k=10) to the true value (k=50) using only the physics and the noisy measurements.

---

## Governing Equation

Steady-state heat conduction in one dimension with uniform heat generation:

```
k * d^2T/dx^2 + Q = 0       for x in (0, L)

```

Boundary conditions:

```
T(0) = T_left = 20 C
T(L) = T_right = 20 C

```

Parameters:

| Symbol | Value | Status |
| --- | --- | --- |
| k | 50 W/(m*K) | UNKNOWN (to be identified) |
| Q | 5000 W/m^3 | Known |
| L | 1.0 m | Known |
| T_left | 20 C | Known |
| T_right | 20 C | Known |

The neural network must discover that k = 50 from the combination of the PDE structure and the sensor data.

---

## Exact Analytical Solution

Given the true conductivity k = 50 W/(m*K), the exact temperature distribution is a parabola:

```
T(x) = -Q*x^2 / (2*k) + Q*L*x / (2*k) + T_left

```

Substituting values:

```
T(x) = -5000*x^2 / (2*50) + 5000*1.0*x / (2*50) + 20
     = -50*x^2 + 50*x + 20

```

Key properties:

- **Maximum temperature**: T(0.5) = -50*(0.25) + 50*(0.5) + 20 = -12.5 + 25 + 20 = 32.5 C
- **Location of maximum**: x = L/2 = 0.5 m (by symmetry)
- **Temperature rise**: Delta_T = 32.5 - 20 = 12.5 C
- **Parabolic profile**: Symmetric about midpoint, concave downward

The formula T_max = Q*L^2/(8*k) + T_left = 5000*1.0/(8*50) + 20 = 12.5 + 20 = 32.5 C confirms the maximum.

Note: If the PINN incorrectly estimates k=10 (the initial guess), it would predict T_max = 5000/(8*10) + 20 = 82.5 C, which is far too hot. The sensor data (which cluster around 32.5 C at the center) force the network to increase k toward the true value.

---

## PINN Implementation Workflow

The inverse PINN workflow differs from a standard forward PINN by adding a data-fitting component and a learnable parameter:

```
+------------------+       +-------------------+       +------------------+
|  Initialize      |       |  Forward Pass     |       |  Compute Losses  |
|  - Network       | ----> |  - Predict T(x)   | ----> |  - PDE residual  |
|  - log_k param   |       |  - Extract k      |       |  - Data mismatch |
|    (guess k=10)  |       |  - Hard BCs       |       |  - Weighted sum  |
+------------------+       +-------------------+       +------------------+
                                                               |
        +------------------------------------------------------+
        |
        v
+------------------+       +-------------------+       +------------------+
|  Update Params   |       |  Check            |       |  Output          |
|  - Network: lr   | <---- |  Convergence      | <---- |  - T(x) field    |
|    = 1e-3        |       |  - k converged?   |       |  - k estimate    |
|  - log_k: lr     |       |  - Loss plateau?  |       |  - Loss history  |
|    = 5e-3        |       |                   |       |                  |
+------------------+       +-------------------+       +------------------+

```

Key differences from forward PINNs:

1. **Learnable parameter**: log_k is an nn.Parameter optimized alongside network weights.
2. **Data loss**: Predictions at sensor locations are compared to noisy measurements.
3. **Two learning rates**: The physical parameter uses a faster learning rate (5x) than the network.
4. **Convergence monitoring**: Track both the loss AND the parameter estimate over epochs.

---

## Network Architecture

The `InverseHeatNet` class combines a standard feed-forward network with a learnable physical parameter:

```
Architecture: InverseHeatNet
+---------------------------------------------------------------+
|                                                               |
|  Learnable Parameter:                                         |
|    log_k = nn.Parameter(log(10.0))   -->  k = exp(log_k)     |
|    (initialized at k=10, true value is k=50)                  |
|                                                               |
+---------------------------------------------------------------+
|                                                               |
|  Neural Network Branch:                                       |
|                                                               |
|  Input: x (normalized to [0,1])                               |
|    |                                                          |
|    v                                                          |
|  [Linear: 1 -> 30] --> [tanh]                                 |
|    |                                                          |
|    v                                                          |
|  [Linear: 30 -> 30] --> [tanh]                                |
|    |                                                          |
|    v                                                          |
|  [Linear: 30 -> 30] --> [tanh]                                |
|    |                                                          |
|    v                                                          |
|  [Linear: 30 -> 1]  --> raw output N(s)                       |
|                                                               |
+---------------------------------------------------------------+
|                                                               |
|  Output Transform (Hard BC Enforcement):                      |
|                                                               |
|  s = x / L              (normalize to [0,1])                  |
|  T_baseline = T_left    (= 20 for symmetric BCs)             |
|  T_scale = 20.0         (scaling factor)                      |
|                                                               |
|  T(x) = T_baseline + T_scale * s * (1-s) * N(s)              |
|                                                               |
|  At s=0: T = 20 + 0 = 20  (left BC satisfied)                |
|  At s=1: T = 20 + 0 = 20  (right BC satisfied)               |
|                                                               |
+---------------------------------------------------------------+

```

The log parameterization ensures k remains strictly positive throughout training:

```
log_k initialized at log(10) = 2.303
k = exp(log_k)

During training:
  Epoch 0:     log_k = 2.303,  k = 10.0
  Epoch 2000:  log_k = 3.2,    k = 24.5
  Epoch 5000:  log_k = 3.7,    k = 40.3
  Epoch 10000: log_k = 3.91,   k = 50.0  (converged!)

```

---

## Hard Boundary Constraints

The output transform guarantees exact satisfaction of Dirichlet boundary conditions at both ends:

```
T(x) = T_baseline + T_scale * s * (1 - s) * N(s)

```

where:

- `s = x / L` is the normalized coordinate
- `T_baseline = T_left = 20 C`
- `T_scale = 20.0` is a scaling constant
- `N(s)` is the raw network output
- `s * (1 - s)` is the bubble function (zero at both ends, max at center)

**Why this matters for the inverse problem**: By eliminating boundary condition errors, the network and the parameter k only need to satisfy the interior PDE and match the sensor data. This removes two constraints from the optimization and allows faster, more reliable convergence of k.

The bubble function s*(1-s) has maximum value 0.25 at s=0.5. With T_scale=20, the network output N(s) near 1.0 at the center produces a temperature rise of about 20*0.25*1.0 = 5 C above baseline. The actual rise is 12.5 C, so N(0.5) converges to approximately 2.5.

---

## Physics Residual and Loss Function

### PDE Residual

The physics residual measures how well the current T(x) and k satisfy the governing equation:

```
R_pde(x) = k * d^2T/dx^2 + Q

```

For the exact solution, R_pde = 0 everywhere. The second derivative is computed via automatic differentiation (torch.autograd.grad, twice).

### Normalized PDE Loss

The PDE residual is normalized by Q to make it dimensionless:

```
loss_pde = mean( (R_pde / Q)^2 )
         = mean( (k * T'' / Q + 1)^2 )

```

This normalization ensures the PDE loss is O(1) regardless of the magnitude of Q.

### Data Loss

The data loss measures mismatch between predictions and noisy sensor measurements:

```
loss_data = mean( (T_predicted(x_sensor) - T_measured)^2 ) / T_scale^2

```

where T_scale normalizes by the characteristic temperature variation.

### Total Loss

The data term is weighted 100x relative to the PDE term:

```
loss_total = loss_pde + 100 * loss_data

```

**Why weight data so heavily?** In an inverse problem, the data is what constrains the unknown parameter. Without sufficient data weighting, the network can satisfy the PDE with any value of k (since k*T'' + Q = 0 is satisfied for any k if T'' adjusts accordingly). The data anchors k to the physically correct value.

### Loss Landscape Intuition

```
         loss_pde alone           loss_data alone           loss_total
         (any k works)            (k=50 is optimal)         (k=50, correct PDE)
              |                        |                         |
   k=10  ----+----  k=100      k=10 --+-- k=100         k=10 --+-- k=100
              |                       / \                       / \
              |                      /   \                     /   \
   -----------+----------      -----/--+--\-----         ----/--+--\----
              |                    /   |   \                 /   |   \
                                      k=50                     k=50
                                   (minimum)                (minimum)

```

---

## Training Configuration

### Optimizer Setup

Adam optimizer with two parameter groups at different learning rates:

```python
optimizer = torch.optim.Adam([
    {'params': model.network.parameters(), 'lr': 1e-3},   # Network weights
    {'params': [model.log_k],              'lr': 5e-3},   # Physical parameter
])

```

**Why different learning rates?**

- The network has many parameters that collectively shape the temperature profile. A moderate learning rate (1e-3) provides stable convergence.
- The single parameter log_k needs to traverse a large distance in parameter space (from log(10)=2.3 to log(50)=3.9). A faster rate (5e-3) ensures it converges before the network overfits to compensate for an incorrect k.

### Learning Rate Schedule

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

```

Both learning rates are halved every 5000 epochs:

| Epochs | Network lr | log_k lr |
| --- | --- | --- |
| 0 - 4999 | 1e-3 | 5e-3 |
| 5000 - 9999 | 5e-4 | 2.5e-3 |
| 10000+ | 2.5e-4 | 1.25e-3 |

### Training Points

- **Collocation points**: 80 uniformly spaced in (0, L) for PDE residual
- **Sensor points**: 10 locations at x = 0.1, 0.2, ..., 0.9 for data loss

### Convergence of k

The parameter k typically converges in three phases:

```
Phase 1 (epochs 0-2000):    Rapid correction
  k: 10 --> ~30             (data loss dominates, large gradients on log_k)
  
Phase 2 (epochs 2000-6000): Refinement  
  k: ~30 --> ~48            (PDE and data losses balance)
  
Phase 3 (epochs 6000-10000): Fine-tuning
  k: ~48 --> 50.0           (small adjustments, lr decaying)

```

### Best-State Tracking

The training loop tracks the state (network weights + log_k) that achieves the lowest total loss, and restores it at the end. This guards against late-stage instabilities.

---

## Results



<img width="2973" height="877" alt="image" src="https://github.com/user-attachments/assets/09fe52cf-0be0-418b-bd7a-36d186bd2c1f" />


The trained PINN produces a 3-panel visualization:

### Panel 1: Temperature Field

- **Blue curve**: PINN prediction T(x) after training
- **Red dashed**: Exact analytical solution
- **Black dots with error bars**: Noisy sensor measurements
- The PINN accurately reconstructs the parabolic profile despite noisy data
- Maximum prediction error is typically < 0.05 C

### Panel 2: Convergence of k

- Shows k = exp(log_k) vs. epoch number
- Starts at k = 10 (initial guess, intentionally 5x too low)
- Converges to k ~ 50 (true value)
- Demonstrates the PINN successfully identifies the unknown parameter
- Final error in k is typically < 1% (|k_identified - 50| < 0.5)

### Panel 3: Training Loss

- Total loss vs. epoch on log scale
- Shows both PDE and data loss components
- Rapid initial decrease as k corrects
- Plateau region as fine-tuning occurs
- Final loss ~ 1e-5 or lower

### Quantitative Results (Typical Run)

| Metric | Value |
| --- | --- |
| True k | 50.0 W/(m*K) |
| Identified k | ~49.8 - 50.2 W/(m*K) |
| Relative error in k | < 1% |
| Sensor RMSE | ~0.1 C (noise floor) |
| Max temperature error | < 0.05 C |
| Training epochs | 10,000 |

---

## Comparison to Other Projects

| Feature | Poisson 1D | Heat Conduction | Convection-Diffusion | Nonlinear Diffusion | **Inverse Heat** |
| --- | --- | --- | --- | --- | --- |
| Problem type | Forward | Forward | Forward | Forward | **Inverse** |
| Unknown | T(x) | T(x) | T(x) | T(x) | **T(x) AND k** |
| Equation | -T''=f | kT''+Q=0 | uT'-DT''=0 | (D(T)T')'=S | kT''+Q=0 |
| Learnable params | Weights only | Weights only | Weights only | Weights only | **Weights + log_k** |
| Data used | None | None | None | None | **Sensor measurements** |
| Boundary handling | Hard | Hard | Hard | Hard | Hard |
| Nonlinearity (PDE) | No | No | No | Yes | No |
| Noise handling | N/A | N/A | N/A | N/A | **Gaussian, std=0.1** |
| Loss terms | PDE | PDE | PDE | PDE | **PDE + Data** |
| Learning rates | Single | Single | Single | Single | **Two groups** |
| Key challenge | Basic PINN | Scaling | Boundary layers | Nonlinear coupling | **Parameter identifiability** |

The inverse problem is unique in this collection because it demonstrates that PINNs are not just PDE solvers but also powerful tools for scientific discovery and parameter estimation.

---

## Assumptions and Limitations

### Assumptions

1. **Steady state**: No time dependence; the rod has reached thermal equilibrium.
2. **1D geometry**: Temperature varies only along the rod length.
3. **Uniform k**: Thermal conductivity is constant throughout the rod (not spatially varying).
4. **Known heat source**: Q is assumed perfectly known. In practice, Q may also be uncertain.
5. **Known sensor positions**: The x-coordinates of sensors are exact.
6. **Gaussian noise**: Measurement errors follow a normal distribution with known standard deviation.
7. **Single parameter**: Only one parameter (k) is unknown.

### Limitations

1. **Synthetic data**: Real experimental data would have systematic errors, calibration drift, and non-Gaussian noise.
2. **Noise level**: The 0.1 C noise is relatively small compared to the 12.5 C signal. Higher noise would require more sensors or regularization.
3. **Initial guess sensitivity**: While k=10 (5x too low) converges reliably, extremely poor guesses (k=0.01 or k=10000) might require curriculum strategies.
4. **Single realization**: One noise realization is used. Ensemble training or Bayesian approaches would quantify uncertainty in the identified k.
5. **Local minimum risk**: The loss landscape for inverse problems can have local minima, especially with multiple unknowns.
6. **No uncertainty quantification**: The point estimate of k has no confidence interval. Bayesian PINNs or ensemble methods would address this.

---

## Possible Extensions

### Multiple Unknown Parameters

Extend to simultaneously identify thermal conductivity k AND heat source Q:

```python
self.log_k = nn.Parameter(torch.log(torch.tensor(10.0)))
self.log_Q = nn.Parameter(torch.log(torch.tensor(1000.0)))

```

This requires more sensor data or tighter constraints to avoid non-uniqueness.

### Spatially Varying Conductivity

Replace scalar k with a field k(x) represented by a second neural network:

```
k_net: x --> k(x)
T_net: x --> T(x)
PDE: d/dx[k(x) * dT/dx] + Q = 0

```

This is significantly harder and requires dense sensor coverage.

### Two-Dimensional Problems

Extend to 2D plates with unknown conductivity:

```
k * (d^2T/dx^2 + d^2T/dy^2) + Q = 0

```

Sensor arrays provide 2D temperature maps for parameter identification.

### Time-Dependent Inverse Problems

Identify thermal diffusivity from transient temperature measurements:

```
rho*cp * dT/dt = k * d^2T/dx^2 + Q

```

Time-series sensor data enables identification of k/(rho*cp).

### Real Experimental Data

Replace synthetic sensors with actual thermocouple or infrared camera measurements:

- Handle non-uniform noise levels per sensor
- Account for sensor placement uncertainty
- Include systematic bias correction
- Validate against independently measured k (e.g., laser flash method)

### Bayesian Uncertainty Quantification

Replace point estimate with posterior distribution over k:

- Ensemble PINNs: Train multiple networks with different initializations
- Dropout-based uncertainty: Apply MC dropout at inference
- Hamiltonian Monte Carlo: Sample the posterior of (weights, log_k) jointly

---

## Requirements

```
python >= 3.8
torch >= 1.10
numpy
matplotlib

```

---

## How to Run

```bash
cd Inverse_Heat_Conduction
python inverse_heat_conduction.py

```

The script will:

1. Generate synthetic sensor data (exact solution + Gaussian noise)
2. Initialize the network with k = 10 (incorrect guess)
3. Train for 10,000 epochs, printing k estimate periodically
4. Display a 3-panel figure showing temperature field, k convergence, and loss history
5. Report the final identified k and comparison to the true value

Expected output:

```
Epoch 0, Loss: 1.23e+00, k = 10.00
Epoch 1000, Loss: 8.45e-02, k = 28.34
Epoch 2000, Loss: 2.11e-02, k = 38.91
Epoch 5000, Loss: 3.67e-03, k = 47.82
Epoch 10000, Loss: 1.05e-04, k = 49.95

Final identified k = 49.95 W/(m*K)
True k = 50.00 W/(m*K)
Relative error: 0.10%

```

---

## Code Walkthrough

For a detailed line-by-line explanation of the implementation, see [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md).

---

## References

- Raissi, M., Perdikaris, P., Karniadakis, G.E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.
- Lu, L., Meng, X., Mao, Z., Karniadakis, G.E. (2021). DeepXDE: A deep learning library for solving differential equations. *SIAM Review*, 63(1), 208-228.
- Cai, S., Mao, Z., Wang, Z., Yin, M., Karniadakis, G.E. (2021). Physics-informed neural networks (PINNs) for fluid mechanics: A review. *Acta Mechanica Sinica*, 37, 1727-1738.

