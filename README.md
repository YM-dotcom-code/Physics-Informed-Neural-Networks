
# Physics-Informed Neural Networks (PINNs)

A repository dedicated to solving partial differential equations (PDEs) using neural networks constrained by physical laws. This is not a single project. It is a growing collection of implementations across different engineering domains, serving as an entry point to understanding and applying PINNs before diving into specific problems.

---

## Table of Contents

- [What This Repository Is About](#what-this-repository-is-about)
- [Background: How We Traditionally Solve PDEs](#background-how-we-traditionally-solve-pdes)
- [What Is a Physics-Informed Neural Network](#what-is-a-physics-informed-neural-network)
- [Key Terminology](#key-terminology)
- [The Core Idea](#the-core-idea)
- [How It Works Step by Step](#how-it-works-step-by-step)
- [The Loss Function Explained](#the-loss-function-explained)
- [Forward vs Inverse Problems](#forward-vs-inverse-problems)
- [When to Use PINNs vs Traditional Solvers](#when-to-use-pinns-vs-traditional-solvers)
- [Sampling Strategies](#sampling-strategies)
- [Known Challenges and Limitations](#known-challenges-and-limitations)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [License](#license)

---

## What This Repository Is About

This repository contains implementations of Physics-Informed Neural Networks applied to various engineering and scientific problems. New projects will be added over time, each solving a different physical system. The domains covered include heat transfer, solid mechanics, fluid dynamics, and coupled multi-physics systems.

Each project is self-contained with its own training scripts, problem definition, and documentation. This README serves as the foundational reference that explains what PINNs are, how they work, and why they matter, so that anyone can understand the approach before exploring individual projects.

---

## Background: How We Traditionally Solve PDEs

In engineering, physical systems are described by partial differential equations. The temperature distribution in a solid, the stress field in a loaded structure, the velocity profile in a fluid, all governed by PDEs.

Traditional numerical methods solve these equations by:

1. Discretizing the domain into a mesh (a grid of elements or cells)
2. Approximating the continuous solution with values at discrete nodes
3. Assembling a system of algebraic equations from the PDE
4. Solving the algebraic system (matrix inversion, iterative solvers)

The **Finite Element Method (FEM)** uses the weak form of the PDE and piecewise polynomial basis functions over elements. Tools like ABAQUS, ANSYS, and COMSOL implement this.

The **Finite Difference Method (FDM)** approximates derivatives using Taylor series expansions on structured grids.

The **Finite Volume Method (FVM)** integrates conservation laws over control volumes. Tools like OpenFOAM and FLUENT use this for CFD.

These methods are powerful and well-established. But they all require a mesh, and that mesh comes with costs: generation time for complex geometries, convergence studies to ensure mesh independence, remeshing for moving boundaries, and entirely separate formulations for inverse problems.

---

## What Is a Physics-Informed Neural Network

A Physics-Informed Neural Network is a neural network that learns to approximate the solution of a PDE by embedding the governing equations directly into its training process.

Instead of learning from labeled input-output pairs (as in standard machine learning), the network is trained by minimizing how much it violates the known physics. The PDE itself becomes the teacher.

The network takes spatial coordinates and time as input, outputs the physical field of interest, and is penalized during training whenever its output fails to satisfy the governing equation, the boundary conditions, or the initial conditions.

The result is a continuous, differentiable function that approximates the solution everywhere in the domain, not just at mesh nodes.

---

## Key Terminology

Before going further, here are the essential terms used throughout this repository:

**PDE (Partial Differential Equation):** An equation involving partial derivatives of an unknown function with respect to multiple variables. Examples: heat equation, wave equation, Navier-Stokes equations.

**Domain (Ω):** The spatial region where the PDE is defined. Could be a 1D bar, a 2D plate, a 3D volume.

**Boundary (∂Ω):** The edges/surfaces of the domain where boundary conditions are applied.

**Collocation Points:** Randomly sampled points inside the domain where the PDE residual is evaluated during training. These replace the mesh nodes of traditional methods.

**Residual:** The amount by which the network's prediction violates the governing equation at a given point. If the network perfectly satisfies the PDE, the residual is zero.

**Automatic Differentiation (Autograd):** A computational technique that computes exact derivatives of the network output with respect to its inputs by traversing the computational graph. This is not finite differences. It produces derivatives exact to machine precision.

**Loss Function:** The objective function minimized during training. In PINNs, it combines PDE residual error, boundary condition error, initial condition error, and optionally data mismatch error.

**Boundary Conditions (BCs):** Constraints on the solution at the domain edges.
- *Dirichlet BC:* fixes the value (e.g., temperature = 100°C at the wall)
- *Neumann BC:* fixes the gradient/flux (e.g., heat flux = 0 at an insulated surface)
- *Robin BC:* a linear combination of value and gradient (e.g., convective heat transfer)

**Initial Conditions (ICs):** The known state of the system at time t = 0 for time-dependent problems.

**Forward Problem:** All equations, parameters, and conditions are known. Find the solution field.

**Inverse Problem:** Some parameters are unknown. Given partial observations, identify both the solution and the unknown parameters.

**Weights (θ):** The trainable parameters of the neural network that are adjusted during training to minimize the loss.

**Epoch:** One complete pass through the training loop (forward pass, loss computation, backpropagation, weight update).

---

## The Core Idea

Consider a general PDE:

$$\mathcal{N}[u(x,t)] = 0, \quad x \in \Omega, \quad t \in [0, T]$$

with boundary conditions:

$$\mathcal{B}[u(x,t)] = 0, \quad x \in \partial\Omega$$

and initial conditions:

$$u(x, 0) = u_0(x), \quad x \in \Omega$$

In a PINN, you replace the unknown field $u(x,t)$ with a neural network $\hat{u}(x,t;\theta)$. The network takes coordinates as input and outputs the predicted field value. You then:

1. Compute the PDE residual by substituting the network output and its derivatives (obtained via autograd) into the governing equation
2. Evaluate boundary and initial condition errors
3. Sum all errors into a total loss
4. Use gradient descent to update the network weights until the loss converges

The total loss function:

$$\mathcal{L}(\theta) = \lambda_{pde} \cdot \mathcal{L}_{pde} + \lambda_{bc} \cdot \mathcal{L}_{bc} + \lambda_{ic} \cdot \mathcal{L}_{ic} + \lambda_{data} \cdot \mathcal{L}_{data}$$

Each term:

$$\mathcal{L}_{pde} = \frac{1}{N_r} \sum_{i=1}^{N_r} \left| \mathcal{N}[\hat{u}(x_r^i, t_r^i; \theta)] \right|^2$$

$$\mathcal{L}_{bc} = \frac{1}{N_b} \sum_{i=1}^{N_b} \left| \mathcal{B}[\hat{u}(x_b^i, t_b^i; \theta)] \right|^2$$

$$\mathcal{L}_{ic} = \frac{1}{N_i} \sum_{i=1}^{N_i} \left| \hat{u}(x_i^i, 0; \theta) - u_0(x_i^i) \right|^2$$

$$\mathcal{L}_{data} = \frac{1}{N_d} \sum_{i=1}^{N_d} \left| \hat{u}(x_d^i, t_d^i; \theta) - u_{obs}^i \right|^2$$

The $\lambda$ coefficients control how much weight each term carries in the total loss.

---

## How It Works Step by Step

**Step 1: Define the problem.**
Write down the governing PDE, boundary conditions, initial conditions, and the computational domain. This is the same starting point as any FEM simulation.

**Step 2: Build the network.**
Construct a fully connected neural network. Inputs are the spatial coordinates (x, y, z) and time (t). Outputs are the physical quantities (temperature, displacement, velocity, pressure). Typical architecture: 4 to 8 hidden layers, 32 to 256 neurons per layer, tanh or sin activation functions.

**Step 3: Sample collocation points.**
Randomly generate points in three categories:
- Interior points (inside Ω) for evaluating the PDE residual
- Boundary points (on ∂Ω) for enforcing boundary conditions
- Initial points (at t = 0) for enforcing initial conditions
- Data points (if available) for matching experimental observations

**Step 4: Forward pass.**
Pass all collocation points through the network to get predictions. Use automatic differentiation to compute all required partial derivatives of the output with respect to the inputs (∂û/∂t, ∂û/∂x, ∂²û/∂x², etc.).

**Step 5: Compute the loss.**
Substitute the network output and its derivatives into the PDE to get the residual at each interior point. Compute boundary and initial condition errors at their respective points. Combine into the total weighted loss.

**Step 6: Backpropagate and update.**
Compute gradients of the total loss with respect to all network weights. Update the weights using an optimizer. Common strategy: Adam optimizer for the first several thousand epochs (fast initial convergence), then switch to L-BFGS (slower but more precise fine-tuning).

**Step 7: Iterate.**
Repeat steps 4-6. Optionally resample collocation points every N epochs to improve coverage and prevent overfitting to specific locations.

**Step 8: Use the trained network.**
Once converged, the network is a continuous function that approximates the solution. Query it at any point in the domain to get the field value instantly. No interpolation needed.

---

## The Loss Function Explained

The loss function is the heart of a PINN. Each component serves a specific purpose:

**PDE Residual Loss** ensures the network respects the governing physics. At every interior collocation point, the PDE is evaluated using the network's output and its autograd-computed derivatives. The mean squared residual is what gets minimized. If this loss is zero, the network perfectly satisfies the PDE everywhere it was sampled.

**Boundary Condition Loss** ensures the solution behaves correctly at the domain edges. For a Dirichlet condition (fixed value), the network output at boundary points must match the prescribed value. For a Neumann condition (fixed gradient), the spatial derivative at boundary points must match the prescribed flux.

**Initial Condition Loss** ensures the solution starts from the correct state. At all points sampled at t = 0, the network output must match the known initial field.

**Data Loss** (optional) ensures agreement with real-world measurements. When sensor data or experimental observations exist, the network prediction at those locations must match the observed values. This is what enables inverse problems and data assimilation.

**Weighting coefficients (λ)** control the relative importance of each term. Getting these right is one of the practical challenges of PINNs. If the PDE loss dominates, the network may ignore boundary conditions. If the BC loss dominates, the interior solution may be inaccurate. Adaptive weighting strategies exist to handle this automatically.

---

## Forward vs Inverse Problems

### Forward Problem

You know everything about the system: the governing PDE, all material parameters, boundary conditions, and initial conditions. You want to find the solution field.

Example: Given thermal conductivity k = 50 W/mK, density ρ = 7800 kg/m³, and specific heat cₚ = 500 J/kgK, solve the heat equation for the temperature field T(x,t) in a steel bar with one end held at 100°C and the other insulated.

The PINN is trained using only the PDE residual, BC, and IC losses. No data is needed.

### Inverse Problem

You know the PDE structure but some parameters are unknown. You have partial measurements of the solution from sensors or experiments. You want both the solution field and the unknown parameters.

Example: You have temperature readings from 5 thermocouples embedded in a bar. You know the heat equation governs the system, but you do not know the thermal conductivity k. Find k.

In the PINN framework, k is simply declared as an additional trainable variable alongside the network weights θ. The data loss term penalizes mismatch with the thermocouple readings. The optimizer adjusts both the network weights and k simultaneously until everything is consistent. No separate optimization loop. No adjoint method. One unified training process.

This is where PINNs truly differentiate themselves from traditional solvers.

---

## When to Use PINNs vs Traditional Solvers

| Scenario | Better Choice | Why |
|:---------|:--------------|:----|
| Standard forward problem, simple geometry | FEM/FDM | Faster, more mature, guaranteed convergence |
| Inverse parameter identification | PINNs | Native capability, no wrapper optimization needed |
| Complex geometry, hard to mesh | PINNs | No mesh required |
| Multi-physics coupling (thermal + structural + fluid) | PINNs | Single unified loss function |
| Parametric studies (many configurations) | PINNs | Train once, predict instantly for new parameters |
| Sparse experimental data + known physics | PINNs | Natural data-physics fusion |
| Need guaranteed error bounds | FEM | Established convergence theory |
| Large-scale industrial simulation | FEM | Decades of solver optimization |
| Sharp discontinuities, fracture, contact | FEM | PINNs struggle with non-smooth solutions |
| Real-time prediction after training | PINNs | Millisecond inference vs minutes of solving |

PINNs do not replace FEM. They are a complementary tool that excels in scenarios where traditional methods are expensive, impractical, or require significant reformulation.

---

## Sampling Strategies

The quality and distribution of collocation points directly affects training convergence and solution accuracy.

**Uniform Random Sampling:** Points drawn uniformly across the domain. Simple to implement. Can miss important regions or cluster in unimportant areas by chance.

**Latin Hypercube Sampling (LHS):** A stratified sampling method that ensures better space-filling coverage. Each dimension is divided into equal intervals, and exactly one sample is placed in each interval. Reduces gaps and clustering compared to pure random sampling.

**Adaptive Residual-Based Sampling:** After initial training, evaluate the PDE residual across the domain. Place additional points in regions where the residual is highest (where the network is struggling most). This focuses computational effort where it matters.

**Resampling:** Regenerate collocation points every N epochs during training. Prevents the network from memorizing specific point locations rather than learning the true continuous solution.

**Boundary-weighted sampling:** Allocate proportionally more points near boundaries and interfaces where gradients tend to be steeper and accuracy matters most.

---

## Known Challenges and Limitations

These are real practical issues you will encounter:

**1. Spectral Bias.**
Neural networks with standard activations (tanh, ReLU) preferentially learn low-frequency components of the solution and struggle with high-frequency content. A smooth, slowly varying temperature field is easy. A rapidly oscillating wave is hard. Mitigation: Fourier feature embeddings, periodic activation functions (sin), or multi-scale architectures.

**2. Loss Balancing.**
The four loss terms often have vastly different magnitudes, especially early in training. If one term dominates the gradient, the others are effectively ignored. Mitigation: adaptive weighting (learning rate annealing per term, gradient normalization, self-attention-based weighting).

**3. Training Cost.**
For a problem where ANSYS gives you an answer in 30 seconds, a PINN might take 30 minutes to train. The payoff comes when: (a) the problem is expensive even for FEM, (b) you need to solve many variations, or (c) you are solving an inverse problem that would require many FEM runs.

**4. Sharp Gradients and Discontinuities.**
Shock waves, material interfaces, contact surfaces, and crack tips produce sharp gradients or discontinuities that smooth neural networks cannot represent well. Mitigation: domain decomposition methods (XPINNs, cPINNs), hard constraint enforcement, or enrichment strategies.

**5. Scalability.**
High-dimensional problems (3D geometry + time + multiple coupled fields) require large networks and many collocation points. GPU memory and training time scale accordingly. Active research area with ongoing improvements.

**6. Hyperparameter Sensitivity.**
Network depth, width, activation function, learning rate, number of collocation points, loss weights, and optimizer choice all affect convergence. There is no universal recipe. Each problem requires some tuning.

---

## Technologies Used

- Python 3.8+
- PyTorch (primary deep learning framework, provides autograd)
- NumPy (numerical operations, array handling)
- Matplotlib (visualization of solutions and training progress)
- SciPy (reference analytical solutions, optimization utilities)

---

## Getting Started

```bash
git clone https://github.com/YM-dotcom-code/Physics-Informed-Neural-Networks.git
cd Physics-Informed-Neural-Networks
pip install torch numpy matplotlib scipy
 ```

## License
This project is licensed under the Apache License 2.0.
