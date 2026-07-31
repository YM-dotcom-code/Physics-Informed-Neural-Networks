# Architecture Comparison

## Physics-Informed Neural Network: Comparing Network Architectures on a Second-Order ODE

This project compares three different neural network architectures solving the same boundary value problem. The key lesson: **architecture choice matters**. A bigger network is not always better, and matching the activation function to the solution character (e.g., sine activations for oscillatory problems) can dramatically improve convergence.

<img width="1614" height="1055" alt="Architecture Comparison Diagram" src="https://github.com/user-attachments/assets/b9700b78-c65e-486b-8dca-a2ea774681fc" />


---

## Table of Contents

- [Background](#background)
- [Key Terminology](#key-terminology)
- [Physical Problem](#physical-problem)
- [Governing Equation](#governing-equation)
- [Analytical Solution](#analytical-solution)
- [PINN Implementation Workflow](#pinn-implementation-workflow)
- [Network Architectures](#network-architectures)
- [Hard Boundary Conditions](#hard-boundary-conditions)
- [Loss Function](#loss-function)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [Comparison to Other Projects](#comparison-to-other-projects)
- [Assumptions and Limitations](#assumptions-and-limitations)
- [Extensions](#extensions)
- [How to Run](#how-to-run)

---

## Background

Physics-Informed Neural Networks (PINNs) embed governing equations directly into the training loss, allowing neural networks to learn solutions that satisfy physical laws. A natural question arises: how does the choice of network architecture - depth, width, and activation function - affect a PINN's ability to learn a given solution?

This project provides a controlled experiment. Three architectures solve the identical ODE with identical training settings. The only variables are network size and activation function. This isolates architecture effects from all other hyperparameter choices.

The problem is deliberately simple (a second-order linear ODE with a known closed-form solution) so that differences in performance can be attributed entirely to architectural choices rather than problem complexity.

---

## Key Terminology

| Term | Definition |
|------|-----------|
| **ODE** | Ordinary Differential Equation - a differential equation containing derivatives with respect to a single independent variable |
| **Dirichlet BC** | A boundary condition that specifies the value of the solution at a boundary point (e.g., u(0) = 0) |
| **Trial solution** | A constructed function form that automatically satisfies boundary conditions, regardless of what the neural network outputs |
| **Activation function** | A nonlinear function applied element-wise after each linear layer in a neural network; determines what function shapes the network can represent |
| **Coordinate transform** | A mapping from one variable range to another (here, s in [0,1] maps to x in [0,pi]) to normalize the input domain |
| **Collocation points** | Discrete points in the domain where the PDE residual is evaluated during training |
| **Residual** | The amount by which the current approximate solution fails to satisfy the governing equation; zero residual means exact satisfaction |
| **Epoch** | One complete pass through all training data (all collocation points) with a weight update |
| **Loss function** | The scalar quantity minimized during training; here, the mean squared PDE residual across all collocation points |

---

## Physical Problem

We solve a second-order ordinary differential equation on a finite interval with homogeneous (zero-value) boundary conditions at both endpoints:

```
u''(x) = -sin(x),    x in [0, pi]
u(0) = 0,  u(pi) = 0
```

**Physical intuition:** This ODE describes a system where the second derivative of u (which represents curvature) is forced by -sin(x). The negative sine forcing pushes the solution into a positive arch shape. Since sin(x) is positive on [0, pi], the forcing -sin(x) is negative, meaning the curvature u''(x) is negative everywhere in the interior - the solution curves downward from any tangent line. Combined with zero boundary conditions at both ends, the solution must rise from zero, arch upward, and return to zero - exactly the shape of sin(x).

The boundary conditions constrain the solution to vanish at both endpoints. This is a classic Dirichlet boundary value problem - the simplest setting for testing PINN architectures because the exact solution is smooth, bounded, and has a single arch shape with no sharp gradients or oscillations.

---

## Governing Equation

The strong form of the boundary value problem:

```
Differential equation:  u''(x) + sin(x) = 0,   x in (0, pi)
Left boundary:          u(0) = 0
Right boundary:         u(pi) = 0
```

This is a linear, second-order ODE with constant coefficients (the coefficient of u'' is 1) and a smooth forcing function sin(x). Linearity means superposition applies, and the smooth forcing guarantees a smooth solution.

---

## Analytical Solution

We solve by direct integration (the standard method for equations of the form u'' = f(x)).

**Step 1: First integration**

```
u''(x) = -sin(x)
```

Integrate both sides with respect to x:

```
u'(x) = cos(x) + C1
```

where C1 is the first constant of integration.

**Step 2: Second integration**

Integrate again:

```
u(x) = sin(x) + C1*x + C2
```

where C2 is the second constant of integration.

**Step 3: Apply boundary condition u(0) = 0**

```
u(0) = sin(0) + C1*(0) + C2 = 0
0 + 0 + C2 = 0
C2 = 0
```

**Step 4: Apply boundary condition u(pi) = 0**

```
u(pi) = sin(pi) + C1*pi + 0 = 0
0 + C1*pi = 0
C1 = 0
```

**Result:**

```
u(x) = sin(x)
```

**Verification:** We confirm the solution satisfies both the ODE and boundary conditions:

| Check | Computation | Result |
|-------|------------|--------|
| u''(x) = -sin(x)? | d^2/dx^2 [sin(x)] = -sin(x) | Yes |
| u(0) = 0? | sin(0) = 0 | Yes |
| u(pi) = 0? | sin(pi) = 0 | Yes |

**Solution values at selected points:**

| x | x/pi | u(x) = sin(x) |
|---|------|---------------|
| 0 | 0.0 | 0.0000 |
| pi/4 | 0.25 | 0.7071 |
| pi/2 | 0.50 | 1.0000 |
| 3*pi/4 | 0.75 | 0.7071 |
| pi | 1.0 | 0.0000 |

The solution is a single smooth arch, symmetric about x = pi/2, with maximum value 1.0 at the midpoint. This shape arises because the forcing function -sin(x) creates exactly the right curvature pattern to produce another sine function - a special property of sine being an eigenfunction of the second derivative operator.

---

## PINN Implementation Workflow

```
1. Define domain:        s in [0, 1]  (normalized coordinate)
2. Coordinate transform: x = pi * s   (physical coordinate)
3. Sample collocation:   60 uniformly spaced points in [0, 1]
4. Forward pass:         s -> Neural Network -> N(s)
5. Apply trial solution: u(s) = 4*s*(1-s)*N(s)
6. Compute derivatives:  u_s, u_ss via automatic differentiation
7. Form residual:        R = u_ss / pi^2 + sin(pi*s)
8. Compute loss:         L = mean(R^2)
9. Backpropagate:        Update weights via Adam optimizer
10. Repeat:             5000 epochs or until loss < 1e-5
```

---

## Network Architectures

Three architectures are compared, all solving the identical problem with identical training:

### PINN_A: Small Tanh Network

```
Architecture: [1, 20, 20, 1]
Activation:   tanh
Parameters:   481
```

- Input layer: 1 neuron (s coordinate)
- Hidden layer 1: 20 neurons with tanh activation
- Hidden layer 2: 20 neurons with tanh activation
- Output layer: 1 neuron (raw network output N(s))
- Parameter count: (1*20 + 20) + (20*20 + 20) + (20*1 + 1) = 20 + 20 + 400 + 20 + 20 + 1 = 481

This is the baseline - a minimal network to see if a small architecture suffices for a smooth solution.

### PINN_B: Wide and Deep Tanh Network

```
Architecture: [1, 50, 50, 50, 1]
Activation:   tanh
Parameters:   5201
```

- Input layer: 1 neuron
- Hidden layer 1: 50 neurons with tanh activation
- Hidden layer 2: 50 neurons with tanh activation
- Hidden layer 3: 50 neurons with tanh activation
- Output layer: 1 neuron
- Parameter count: (1*50 + 50) + (50*50 + 50) + (50*50 + 50) + (50*1 + 1) = 50 + 50 + 2500 + 50 + 2500 + 50 + 50 + 1 = 5201

This is over 10x larger than PINN_A. The question: does 10x more parameters mean 10x better performance?

### PINN_C: Small Sine Activation Network

```
Architecture: [1, 20, 20, 1]
Activation:   sin (custom SinActivation class)
Parameters:   481
```

- Identical structure to PINN_A (same width, depth, parameter count)
- Only difference: replaces tanh with sin(x) as the activation function
- Uses a custom `SinActivation` class that applies torch.sin element-wise

The hypothesis: since the exact solution is sin(x), a network whose basis functions are compositions of sine functions should represent it more naturally than tanh-based networks.

**Why sine activations help for this problem:** Standard tanh activations approximate smooth functions by combining sigmoid-like shapes. To represent a sinusoidal function, many tanh neurons must cooperate. A sine activation, by contrast, can represent sin(x) with a single neuron given the right weight and bias. For periodic or oscillatory target functions, this built-in periodicity is a structural advantage.

---

## Hard Boundary Conditions

Instead of penalizing boundary violations in the loss function (soft enforcement), this implementation uses a **trial solution** that makes boundary satisfaction automatic:

```
u(s) = 4 * s * (1 - s) * N(s)
```

where N(s) is the raw neural network output.

**Why this works:**

- At s = 0: u(0) = 4 * 0 * (1-0) * N(0) = 0, regardless of N(0)
- At s = 1: u(1) = 4 * 1 * (1-1) * N(1) = 0, regardless of N(1)

The factor s*(1-s) is zero at both boundaries, so the product is zero no matter what the network outputs. The constant 4 scales the envelope so that the maximum of s*(1-s) (which occurs at s=0.5 with value 0.25) becomes 1.0, giving the network output N(s) direct control over the solution magnitude at the midpoint.

**Advantage of hard BCs:** The loss function only needs to enforce the PDE residual. There is no balancing act between boundary loss and interior loss. The optimizer focuses entirely on satisfying the differential equation.

---

## Loss Function

The loss is the mean squared PDE residual evaluated at the collocation points:

```
Loss = (1/N) * sum_{i=1}^{N} R(s_i)^2
```

where the residual at each point is:

```
R(s) = u_ss(s) / pi^2 + sin(pi * s)
```

**Origin of the pi^2 factor:** The coordinate transform x = pi*s means:

```
du/dx = (du/ds) * (ds/dx) = (1/pi) * du/ds
d^2u/dx^2 = (1/pi^2) * d^2u/ds^2
```

So the ODE u''(x) + sin(x) = 0 becomes:

```
(1/pi^2) * u_ss + sin(pi*s) = 0
```

The second derivative u_ss is computed via PyTorch automatic differentiation (two successive calls to `torch.autograd.grad` with `create_graph=True` to allow backpropagation through the derivative computation).

---

## Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Collocation points | 60 | Sufficient for a smooth single-arch solution on [0,1] |
| Optimizer | Adam | Adaptive learning rates handle varying gradient magnitudes |
| Initial learning rate | 0.001 | Standard Adam default; works well for most PINN problems |
| Scheduler | StepLR | Halves learning rate at epoch 2500 for fine-tuning |
| Total epochs | 5000 | Generous budget; early stopping at loss < 1e-5 |
| Convergence criterion | loss < 1e-5 | Ensures PDE residual is negligibly small |
| Best-state tracking | Yes | Saves model weights at lowest loss encountered |

**Training strategy:** The learning rate starts at 0.001 for rapid initial descent, then halves at epoch 2500 to allow fine convergence. Best-state tracking ensures that if the loss temporarily increases (common with Adam near convergence), the best weights are preserved.

---

## Results

<img width="2780" height="977" alt="image" src="https://github.com/user-attachments/assets/3ee4db37-7176-4720-9414-3f7f141e60a7" />

The output figure contains two panels:

**Left panel - Solution predictions:**
All three architectures overlaid on the exact solution sin(x). This shows how closely each network approximates the true solution after training.

**Right panel - Convergence curves:**
Log-scale loss vs. epoch for all three architectures. This reveals:

- **Convergence speed:** How quickly each architecture reduces the PDE residual
- **Final accuracy:** The lowest loss achieved
- **Training stability:** Whether the loss decreases smoothly or oscillates

**Key observations:**

1. **PINN_A (small tanh):** Converges reliably but may require most of the 5000 epochs. The small network has enough capacity for this smooth solution.

2. **PINN_B (wide/deep tanh):** Despite having 10x more parameters, does NOT converge 10x faster. The extra capacity provides no benefit for this simple problem and may even slow convergence due to the larger parameter space the optimizer must navigate.

3. **PINN_C (small sine):** Converges fastest. The sine activation function provides a natural basis for representing the sinusoidal solution. The network can represent sin(pi*s) almost exactly with minimal weight tuning.

**The key lesson:** For this sinusoidal problem, matching the activation function to the solution character (sine activations for a sine solution) outperforms brute-force capacity increases. Architecture design should be informed by the expected solution behavior.

---

## Comparison to Other Projects

This architecture comparison study sits within a broader collection of PINN projects, each demonstrating different aspects of the methodology:

| Project | Physics | Equation Type | Key PINN Feature |
|---------|---------|--------------|-----------------|
| **Architecture Comparison** (this) | Simple ODE forcing | 2nd-order ODE | Architecture effects, activation functions |
| Thermoelastic Bar | Coupled thermal-mechanical | Coupled ODEs | Multi-physics coupling, temperature-dependent properties |
| Channel Flow | Viscous fluid between plates | 2nd-order ODE (Navier-Stokes) | Pressure gradient forcing, physical parameter recovery |
| Heat Source | Steady conduction with generation | 2nd-order ODE | Source term identification, known solution verification |
| Cantilever Beam | Structural deflection under load | 4th-order ODE | Higher-order derivatives, multiple BCs (clamped + free) |
| Inverse Heat | Parameter estimation from data | 2nd-order ODE (inverse) | Inverse problem formulation, data-driven parameter recovery |

**What makes this project unique:** While other projects focus on increasingly complex physics, this project holds the physics constant and varies the neural network architecture. It answers the question: "Given a fixed problem, how much does architecture choice matter?" The answer is: substantially.

**Progression suggestion:** Start with this project to understand architecture effects, then apply those insights when choosing architectures for the more complex problems above.

---

## Assumptions and Limitations

**Assumptions:**

- The ODE is linear with smooth coefficients and smooth forcing - the simplest possible test case
- All three architectures use identical training hyperparameters (learning rate, scheduler, epochs) - a fair comparison but possibly not optimal for each individual architecture
- 60 collocation points are uniformly spaced - no adaptive refinement
- The coordinate transform to [0,1] is fixed and identical for all architectures
- The trial solution form 4*s*(1-s)*N(s) works because both BCs are homogeneous Dirichlet

**Limitations:**

- Results may not generalize to problems with sharp gradients, discontinuities, or high-frequency content
- The comparison uses a single random initialization - results could vary across seeds
- Only two activation functions are tested (tanh and sine) - others (ReLU, GELU, swish) are not explored
- The "bigger isn't better" conclusion applies to this smooth, low-complexity problem; for PDEs with multiscale features, larger networks may be essential
- No regularization techniques (dropout, weight decay) are applied - these could change the relative rankings

---

## Extensions

Potential directions for extending this study:

1. **More activation functions:** Compare ReLU, GELU, Swish, and learnable activations (e.g., adaptive sine frequency)
2. **Higher-frequency solutions:** Test on u''(x) = -n^2*sin(n*x) for n = 2, 3, 5, 10 to see if sine activation advantage grows with frequency
3. **Adaptive architectures:** Start small and grow the network during training if loss plateaus
4. **Ensemble methods:** Combine predictions from all three architectures for uncertainty quantification
5. **Initialization study:** Run multiple random seeds and report statistics (mean, std of final loss)
6. **Residual-based architecture search:** Use the spatial distribution of residual errors to guide where more capacity is needed
7. **Transfer learning:** Train on one ODE, then fine-tune on a different forcing function to test architecture flexibility
8. **Computational cost:** Compare not just final accuracy but wall-clock time and floating-point operations per epoch

---

## How to Run

**Prerequisites:**

```
Python 3.8+
PyTorch 1.9+
NumPy
Matplotlib
```

**Execution:**

```bash
cd Architecture_Comparison
python architecture_comparison.py
```

**Output:**

- `chapter3_architecture_comparison.png` - Two-panel figure showing solution predictions and convergence curves
- Console output with training progress and final loss values for each architecture

**Expected runtime:** Approximately 1-3 minutes depending on hardware (CPU is sufficient for this problem size).

---

## File Structure

```
Architecture_Comparison/
|- architecture_comparison.py    # Main script: defines all three architectures, trains, and plots
|- chapter3_architecture_comparison.png  # Output: 2-panel results figure
|- arch_comparison_diagram.png   # Architecture diagram
|- README.md                     # This file
```

---

## Summary

This project demonstrates that for physics-informed neural networks, thoughtful architecture selection - particularly matching the activation function to the expected solution character - can outperform naive capacity scaling. A 481-parameter network with sine activations converges faster and more accurately than a 5201-parameter network with tanh activations when the target solution is sinusoidal. This insight should guide architecture choices in all subsequent PINN projects: consider what basis functions your activation provides, and whether they align with the physics you expect to capture.
