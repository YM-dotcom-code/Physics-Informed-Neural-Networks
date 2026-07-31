# Trivial Solution Demo

A Physics-Informed Neural Network (PINN) demonstration showing how non-unique boundary value problems cause PINNs to collapse to the trivial solution, and how adding an amplitude condition fixes the problem.

<img width="1606" height="810" alt="Trivial Solution Diagram" src="https://github.com/user-attachments/assets/ca0cc383-79ab-40dd-8b1a-2671d913bdf7" />


---

## Table of Contents

1. [Background](#background)
2. [Key Terminology](#key-terminology)
3. [Physical Problem](#physical-problem)
4. [Governing Equation](#governing-equation)
5. [Analytical Solution](#analytical-solution)
6. [Why the Problem is Non-Unique](#why-the-problem-is-non-unique)
7. [The PINN Failure Mode](#the-pinn-failure-mode)
8. [The Fix - Adding an Amplitude Condition](#the-fix---adding-an-amplitude-condition)
9. [PINN Implementation Workflow](#pinn-implementation-workflow)
10. [Architecture](#architecture)
11. [Trial Solutions](#trial-solutions)
12. [Loss Function](#loss-function)
13. [Training Configuration](#training-configuration)
14. [Results](#results)
15. [Comparison to Other Projects](#comparison-to-other-projects)
16. [Assumptions and Limitations](#assumptions-and-limitations)
17. [Extensions](#extensions)

---

## Background

Physics-Informed Neural Networks (PINNs) embed differential equations directly into the training loss so that the network learns solutions that satisfy physical laws. Typically, a well-posed boundary value problem (BVP) has a unique solution, and the PINN converges to it. But what happens when the BVP admits infinitely many solutions?

This project demonstrates a critical failure mode: when the boundary value problem does not uniquely determine a solution, the PINN collapses to the trivial solution u = 0. It then shows how to recover the physically meaningful answer by adding a single extra condition that pins the amplitude.

Understanding this failure mode is essential for anyone applying PINNs to eigenvalue problems, resonance phenomena, or any differential equation where boundary conditions alone do not guarantee uniqueness.

---

## Key Terminology

| Term | Definition |
|------|-----------|
| **Eigenvalue problem** | A differential equation of the form L[u] = lambda * u where solutions exist only for special values of lambda (eigenvalues). Here, u'' + u = 0 is an eigenvalue problem with lambda = 1. |
| **Homogeneous BVP** | A boundary value problem where both the equation and boundary conditions equal zero. The zero function always satisfies such a problem. |
| **Trivial solution** | The solution u(x) = 0 for all x. It satisfies any homogeneous BVP automatically without carrying physical meaning. |
| **Non-uniqueness** | The property that more than one function satisfies all the given conditions. Here, u = C*sin(x) works for any constant C. |
| **Well-posedness** | A problem is well-posed (in the sense of Hadamard) if a solution exists, is unique, and depends continuously on the data. Our original BVP fails the uniqueness requirement. |
| **Characteristic equation** | An algebraic equation obtained by substituting u = e^(rx) into a linear ODE with constant coefficients. For u'' + u = 0, the characteristic equation is r^2 + 1 = 0. |
| **Amplitude condition** | An extra constraint (such as u(pi/2) = 1) that fixes the arbitrary constant in a non-unique problem, selecting one specific solution from the family. |
| **Trial solution** | A neural network ansatz (formula) constructed so that boundary conditions are satisfied exactly by design, regardless of what the network outputs. |

---

## Physical Problem

Consider a vibrating string fixed at both endpoints, or equivalently a quantum particle in a box. The spatial part of the wave equation leads to the eigenvalue problem:

- Find u(x) on the interval [0, pi]
- Such that the second derivative of u plus u itself equals zero
- With the string pinned at both ends: u(0) = 0 and u(pi) = 0

This is the simplest eigenvalue problem in mathematical physics. The eigenvalue lambda = 1 corresponds to the fundamental mode of vibration, and the eigenfunction is sin(x) - but only up to an arbitrary multiplicative constant.

---

## Governing Equation

The ordinary differential equation is:

```
u''(x) + u(x) = 0,   x in [0, pi]
```

where u''(x) denotes the second derivative of u with respect to x.

Boundary conditions:

```
u(0) = 0
u(pi) = 0
```

This is a second-order linear homogeneous ODE with constant coefficients, paired with homogeneous boundary conditions.

---

## Analytical Solution

We solve the ODE step by step.

**Step 1 - Form the characteristic equation.**

For a linear ODE with constant coefficients u'' + u = 0, we substitute the trial form u = e^(rx):

```
r^2 * e^(rx) + e^(rx) = 0
e^(rx) * (r^2 + 1) = 0
```

Since e^(rx) is never zero, we require:

```
r^2 + 1 = 0
```

**Step 2 - Solve the characteristic equation.**

```
r^2 = -1
r = +i  or  r = -i
```

where i is the imaginary unit (i^2 = -1).

**Step 3 - Write the general solution.**

Complex exponential roots r = +/- i give oscillatory solutions. Using Euler's formula:

```
u(x) = A*cos(x) + B*sin(x)
```

where A and B are arbitrary constants determined by boundary conditions.

**Step 4 - Apply the first boundary condition u(0) = 0.**

```
u(0) = A*cos(0) + B*sin(0) = A*1 + B*0 = A = 0
```

So A = 0, and the solution reduces to:

```
u(x) = B*sin(x)
```

**Step 5 - Apply the second boundary condition u(pi) = 0.**

```
u(pi) = B*sin(pi) = B*0 = 0
```

This equation is satisfied for ANY value of B. The boundary condition provides no information about B.

**Step 6 - Conclusion.**

The solution is:

```
u(x) = C*sin(x)
```

where C is an arbitrary constant. The problem has infinitely many solutions.

---

## Why the Problem is Non-Unique

The non-uniqueness arises because sin(pi) = 0. The second boundary condition u(pi) = 0 is automatically satisfied by sin(x) at x = pi, regardless of the amplitude. Geometrically, the sine function has a natural zero at x = pi, so pinning the string there adds no constraint.

This is a hallmark of eigenvalue problems at resonance: the differential operator u'' + u has a nontrivial null space (the set of functions satisfying L[u] = 0 with homogeneous BCs is not just {0}). The eigenfunction sin(x) spans this null space, and any scalar multiple remains in it.

In contrast, if the equation were u'' + 4u = 0 (eigenvalue lambda = 4), the general solution would be A*cos(2x) + B*sin(2x), and applying u(0) = 0, u(pi) = 0 would give A = 0 and B*sin(2*pi) = 0, which is again satisfied for any B. But for u'' + 2u = 0 with solution A*cos(sqrt(2)*x) + B*sin(sqrt(2)*x), the boundary conditions would typically force both A = 0 and B = 0 (since sin(sqrt(2)*pi) is not zero), giving only the trivial solution. The non-uniqueness is special to eigenvalue problems.

---

## The PINN Failure Mode

When a PINN is trained on this underdetermined problem, it consistently converges to u(x) = 0 rather than any nontrivial member of the solution family. This happens for three reasons:

**1. Implicit regularization toward small weights.** Neural networks initialized with small random weights produce outputs near zero. The trivial solution u = 0 is the closest valid solution to the network's initial state, so gradient descent finds it first.

**2. Zero is a global minimum of the PDE loss.** The residual u'' + u equals zero when u = 0 everywhere. The network does not need to learn any structure - it simply needs its output to remain near zero, which requires minimal departure from initialization.

**3. No gradient signal pointing toward nontrivial solutions.** Since u = 0 satisfies the PDE and boundary conditions exactly, the loss landscape has zero gradient at this point. Once the network approaches u = 0, there is no force pushing it toward C*sin(x) for any nonzero C. All nontrivial solutions are equally valid but require the network to climb out of the u = 0 basin, which gradient descent will not do.

In summary, the PINN has no way to "know" that a nontrivial solution exists. It sees a loss function that is perfectly minimized by zero, and it stops there.

---

## The Fix - Adding an Amplitude Condition

To make the problem well-posed, we add a normalization condition that selects exactly one solution from the family u = C*sin(x):

```
u(pi/2) = 1
```

Substituting into u(x) = C*sin(x):

```
C*sin(pi/2) = C*1 = 1
```

Therefore C = 1, and the unique solution is:

```
u(x) = sin(x)
```

This single additional constraint eliminates the non-uniqueness entirely. The PINN now has a well-posed target and converges to sin(x) reliably.

The choice of pi/2 is natural because sin(pi/2) = 1 is the maximum of sin(x) on [0, pi], but any interior point where sin is nonzero would work (with appropriate rescaling of the target value).

---

## PINN Implementation Workflow

The implementation in `trivial_solution_demo.py` follows these steps:

1. **Coordinate transformation** - Map the physical domain x in [0, pi] to a computational domain s in [0, 1] for numerical convenience.
2. **Network definition** - Build a feedforward neural network that takes s as input and produces a scalar output N(s).
3. **Trial solution construction** - Wrap the raw network output in an ansatz that enforces boundary conditions exactly.
4. **Collocation point sampling** - Distribute training points across the domain where the PDE residual will be evaluated.
5. **Loss computation** - Compute the PDE residual at collocation points and minimize its mean squared value.
6. **Training loop** - Run Adam optimizer with learning rate scheduling and best-state tracking.
7. **Comparison** - Train both the underdetermined case and the fixed case, then plot results side by side.

---

## Architecture

The neural network uses a fully connected feedforward architecture:

```
Input layer:    1 neuron  (takes s in [0, 1])
Hidden layer 1: 20 neurons, tanh activation
Hidden layer 2: 20 neurons, tanh activation
Hidden layer 3: 20 neurons, tanh activation
Output layer:   1 neuron  (produces raw output N(s))
```

Layer dimensions: [1, 20, 20, 20, 1]

**Why tanh?** The tanh activation function is smooth (infinitely differentiable), which is essential because PINNs require computing second derivatives through the network via automatic differentiation. Non-smooth activations like ReLU would produce zero or undefined second derivatives, making the PDE residual meaningless.

**Why 3 hidden layers with 20 neurons?** This is a relatively small network suitable for a 1D problem. The function sin(x) is simple, so a large network is unnecessary. Three layers provide enough depth for the network to represent smooth oscillatory functions without overfitting the collocation points.

---

## Trial Solutions

The code uses a coordinate transformation from s in [0, 1] to x in [0, pi]. Under this mapping, derivatives transform as:

```
x = pi * s
dx = pi * ds
u_x = u_s / pi
u_xx = u_ss / pi^2
```

So the PDE in the s-coordinate becomes:

```
u_ss / pi^2 + u = 0
```

### Case 1 - Underdetermined (boundary conditions only)

```
u(s) = 4 * s * (1 - s) * N(s)
```

where N(s) is the raw neural network output.

**Why this works for boundary conditions:**
- At s = 0: u = 4 * 0 * 1 * N(0) = 0 (satisfies u(0) = 0)
- At s = 1: u = 4 * 1 * 0 * N(1) = 0 (satisfies u(pi) = 0)

The factor 4 * s * (1 - s) is a "bubble function" that vanishes at both endpoints regardless of N(s). This enforces boundary conditions exactly without any penalty terms.

**The problem:** This trial solution can represent any smooth function that vanishes at the endpoints. It does not select a specific amplitude, so the PINN converges to u = 0.

### Case 2 - With amplitude condition (well-posed)

```
u(s) = 4 * s * (1 - s) * (1 + (s - 0.5) * N(s))
```

**Why this also enforces u(pi/2) = 1:**

At s = 0.5 (which corresponds to x = pi/2):

```
u(0.5) = 4 * 0.5 * 0.5 * (1 + (0.5 - 0.5) * N(0.5))
       = 4 * 0.25 * (1 + 0 * N(0.5))
       = 1 * 1
       = 1
```

The key trick is that the factor (s - 0.5) multiplying N(s) vanishes at s = 0.5. This means the network output at the midpoint is irrelevant - the trial solution evaluates to exactly 1 there, no matter what N produces. Combined with the s*(1-s) envelope enforcing zero at both endpoints, all three conditions are hardcoded into the ansatz.

**Boundary conditions still hold:**
- At s = 0: u = 4 * 0 * 1 * (...) = 0
- At s = 1: u = 4 * 1 * 0 * (...) = 0

This is a beautiful construction: three constraints (two boundary values and one interior value) are all enforced algebraically, leaving the network free to adjust only the shape between these fixed points.

---

## Loss Function

The loss is purely the PDE residual - no boundary penalty terms are needed because the trial solutions enforce boundary conditions exactly.

For both cases, the PDE residual at each collocation point s_i is:

```
R(s_i) = u_ss(s_i) / pi^2 + u(s_i)
```

where u_ss is the second derivative of the trial solution with respect to s, computed via automatic differentiation (torch.autograd.grad with create_graph=True for higher-order derivatives).

The loss function is the mean squared residual over all collocation points:

```
Loss = (1/N) * sum_{i=1}^{N} R(s_i)^2
```

where N = 60 is the number of collocation points.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Collocation points | 60 (uniformly spaced in [0, 1]) |
| Optimizer | Adam |
| Initial learning rate | 0.001 (Adam default) |
| Learning rate schedule | StepLR - halve the learning rate at epoch 2500 |
| Total epochs | 5000 |
| Best-state tracking | Yes - saves network weights at lowest loss |
| Weight initialization | PyTorch default (Kaiming uniform) |

**Why best-state tracking?** The loss may fluctuate during training, especially after the learning rate drop. By saving the network state at the lowest observed loss, we ensure the final prediction uses the most accurate weights found during the entire training run.

**Why StepLR at 2500?** The first half of training explores the loss landscape broadly. Halving the learning rate at the midpoint allows fine-grained convergence in the second half without getting stuck in early training.

---

## Results

<img width="3178" height="877" alt="image" src="https://github.com/user-attachments/assets/f2ac45ad-d216-4d7f-88d7-36564cdd5b74" />


The output is a three-panel figure:

**Panel 1 - Loss convergence curves.** Both cases show decreasing PDE residual over 5000 epochs. The underdetermined case (Case 1) converges quickly to near-zero loss because u = 0 is easy to learn. The well-posed case (Case 2) also converges but takes longer because it must learn the nontrivial shape of sin(x).

**Panel 2 - Underdetermined PINN vs reference.** The PINN output (blue) is essentially zero everywhere, while the reference solution sin(x) (dashed) shows the expected sinusoidal shape. This demonstrates the failure mode: the network finds u = 0 because no condition forces it to find anything else.

**Panel 3 - Fixed PINN vs exact solution.** After adding the amplitude condition u(pi/2) = 1, the PINN (blue) closely matches the exact solution sin(x) (dashed). The network successfully learns the correct shape because the problem is now well-posed.

---

## Comparison to Other Projects

| Aspect | Trivial Solution Demo | Standard PINN (e.g., Poisson) | Eigenvalue PINN |
|--------|----------------------|-------------------------------|-----------------|
| Problem type | Homogeneous eigenvalue BVP | Inhomogeneous BVP | Eigenvalue search |
| Uniqueness | Non-unique without amplitude condition | Unique (forcing term breaks symmetry) | Non-unique (eigenfunction scaling) |
| Failure mode demonstrated | Collapse to u = 0 | None (well-posed by default) | May find wrong eigenvalue |
| Fix applied | Hard-coded amplitude constraint in trial solution | Not needed | Normalization constraint |
| Trial solution complexity | Two variants compared | Single ansatz | Requires eigenvalue as trainable parameter |
| Key insight | Well-posedness is a prerequisite for PINN success | Standard PINN workflow | Must constrain both eigenvalue and eigenfunction |
| Boundary enforcement | Exact (built into trial solution) | Often penalty-based | Varies |
| Domain | 1D, [0, pi] | Typically 1D or 2D | 1D or higher |
| Network size | Small (3x20) | Problem-dependent | Usually larger |

---

## Assumptions and Limitations

**Assumptions:**

1. The coordinate transformation s in [0, 1] -> x in [0, pi] is linear and introduces no numerical issues.
2. Sixty collocation points are sufficient to resolve sin(x) on [0, pi] (the function has no sharp gradients).
3. The tanh network with 3 hidden layers of width 20 has sufficient capacity to represent sin(x) accurately.
4. Adam optimizer with default momentum parameters is appropriate for this smooth loss landscape.

**Limitations:**

1. This demo addresses only the simplest eigenvalue problem. Higher modes (sin(2x), sin(3x), ...) would require different amplitude conditions.
2. The trial solution construction is specific to this problem geometry. Generalizing to 2D or irregular domains requires different ansatz strategies.
3. The code does not explore what happens with different initializations - in rare cases, random initialization might break symmetry and find a nontrivial solution even without the amplitude condition, but this is not reliable.
4. No error quantification (L2 error, pointwise error bounds) is reported beyond visual comparison.
5. The approach assumes we know the location where the amplitude condition should be applied (here pi/2). In practice, choosing this point requires some knowledge of the expected solution.

---

## Extensions

1. **Higher eigenmodes.** Replace the amplitude condition with u(pi/4) = sin(pi/4) or target sin(nx) for higher modes. Explore whether PINNs can distinguish between modes.

2. **Eigenvalue as trainable parameter.** Instead of fixing lambda = 1 in u'' + lambda*u = 0, make lambda a learnable parameter alongside the network weights. The amplitude condition then determines both the eigenvalue and the eigenfunction simultaneously.

3. **2D eigenvalue problems.** Extend to the Laplacian eigenvalue problem on a rectangle or disk, where non-uniqueness manifests as degenerate eigenspaces.

4. **Nonlinear eigenvalue problems.** Consider equations like u'' + u^3 = 0 where the amplitude affects the "eigenvalue" (frequency depends on amplitude). PINNs may behave differently here because the trivial solution is isolated.

5. **Systematic study of initialization.** Investigate how network initialization scale affects the bias toward u = 0. Does Xavier initialization versus small uniform initialization change the outcome?

6. **Loss landscape visualization.** Plot the loss as a function of a single "amplitude" parameter to show that u = 0 is a local (and global) minimum when no amplitude condition is present.

7. **Alternative normalization strategies.** Instead of a pointwise amplitude condition, try integral normalization (integral of u^2 = 1) or maximum-value constraints. Compare effectiveness and training stability.

---

## File Structure

```
Trivial_Solution_Demo/
    trivial_solution_demo.py          # Main training script
    chapter1_trivial_solution_demo.png # Output figure (3-panel comparison)
    README.md                          # This file
```

---

## How to Run

```bash
cd Trivial_Solution_Demo
python trivial_solution_demo.py
```

The script will train both cases sequentially and save the output figure as `chapter1_trivial_solution_demo.png`.

---

## Key Takeaway

Well-posedness is not optional - it is a prerequisite for PINN success. If your boundary value problem admits non-unique solutions, the neural network will almost certainly find the trivial one. Before training, always verify that your problem has a unique solution. If it does not, add normalization conditions, amplitude constraints, or other supplementary equations until the solution is uniquely determined. Only then will the PINN reliably converge to a physically meaningful answer.
