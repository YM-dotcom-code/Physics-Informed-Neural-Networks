# Autograd Verification: Proving PyTorch Computes Exact Derivatives

<img width="1485" height="1035" alt="autograd_diagram" src="https://github.com/user-attachments/assets/b1d46a08-2222-47e6-8f07-3d8653c8d001" />


This project is a foundational demonstration that PyTorch's automatic differentiation engine (autograd) computes mathematically exact derivatives, limited only by floating-point precision. It is not a neural network project and does not solve any PDE. Instead, it proves that the core technology underlying Physics-Informed Neural Networks (PINNs) works correctly, producing errors at machine epsilon (~10^-15) rather than the truncation errors inherent in numerical approximation methods.

---

## Table of Contents

1. [Background](#background)
   - [What Is Automatic Differentiation](#what-is-automatic-differentiation)
   - [How It Differs from Finite Differences](#how-it-differs-from-finite-differences)
   - [How It Differs from Symbolic Differentiation](#how-it-differs-from-symbolic-differentiation)
   - [Why PINNs Need Exact Derivatives](#why-pinns-need-exact-derivatives)
   - [What Is Machine Epsilon](#what-is-machine-epsilon)
2. [Key Terminology](#key-terminology)
3. [What This Script Demonstrates](#what-this-script-demonstrates)
4. [Part 1: Derivative Verification](#part-1-derivative-verification)
   - [The Test Function](#the-test-function)
   - [How Autograd Computes the Derivatives](#how-autograd-computes-the-derivatives)
   - [Results: Errors at Machine Epsilon](#results-errors-at-machine-epsilon)
5. [Part 2: Heat Equation Residual Verification](#part-2-heat-equation-residual-verification)
   - [The Heat Equation and Its Exact Solution](#the-heat-equation-and-its-exact-solution)
   - [Computing Derivatives via Autograd](#computing-derivatives-via-autograd)
   - [Residual Should Be Zero](#residual-should-be-zero)
   - [Results: Residual at Machine Epsilon](#results-residual-at-machine-epsilon)
6. [Why This Matters for PINNs](#why-this-matters-for-pinns)
7. [Results](#results)
8. [Connection to Other Projects](#connection-to-other-projects)
9. [Requirements](#requirements)
10. [How to Run](#how-to-run)

---

## Background

### What Is Automatic Differentiation

Automatic differentiation (autodiff or autograd in PyTorch) is a technique for computing derivatives of functions defined by computer programs. It works by breaking every computation into a sequence of elementary operations (addition, multiplication, sin, exp, etc.) and applying the chain rule systematically through that sequence.

The key insight is that every elementary operation has a known, exact derivative. When you chain these exact derivatives together via the chain rule, the final result is the exact derivative of the overall function, not an approximation.

PyTorch implements this through a system called autograd. When you perform operations on tensors that have `requires_grad=True`, PyTorch records the sequence of operations in a computational graph. When you call `.backward()` or `torch.autograd.grad()`, it traverses this graph in reverse, applying the chain rule at each step to produce the exact derivative.

### How It Differs from Finite Differences

Finite differences approximate a derivative by evaluating the function at nearby points:

```
df/dx ~ [f(x + h) - f(x - h)] / (2h)
```

This introduces truncation error proportional to h^2. If h = 0.01, the error is on the order of 10^-4. Making h smaller helps, but eventually roundoff error from subtracting nearly equal numbers dominates. The best achievable accuracy for finite differences is typically around 10^-8 for first derivatives and worse for higher derivatives.

Autograd has no step size h. It computes the exact derivative through the chain rule. The only error comes from representing numbers in finite-precision floating point, which gives errors around 10^-15 for float64.

### How It Differs from Symbolic Differentiation

Symbolic differentiation (like Mathematica or SymPy) manipulates mathematical expressions algebraically. It applies differentiation rules to produce a new symbolic expression for the derivative. This gives an exact formula, but:

- It requires the function to be expressible as a closed-form formula.
- Expressions can grow exponentially large (expression swell).
- It cannot differentiate through arbitrary program logic (loops, conditionals).
- It must simplify the resulting expression, which is computationally expensive.

Autograd operates on numerical values flowing through a computational graph. It handles any computation that PyTorch can express, including neural networks with millions of parameters, without ever forming a symbolic expression.

### Why PINNs Need Exact Derivatives

Physics-Informed Neural Networks work by embedding physical laws (differential equations) directly into the loss function. For example, if the physics requires u_t = alpha * u_xx, then the PINN loss includes a term like:

```
loss_physics = mean( (u_t - alpha * u_xx)^2 )
```

The network is trained so that its output satisfies this equation. But this only works if u_t and u_xx are computed exactly from the network output. If we used finite differences to estimate u_xx, we would introduce O(h^2) approximation errors that would:

- Corrupt the loss signal, making the network chase approximation artifacts.
- Require extremely fine grids to get acceptable accuracy, defeating the purpose of PINNs.
- Make the residual nonzero even for a perfect solution.

With autograd, derivatives are exact (to machine precision), so the only error in the physics residual comes from the network not yet having learned the solution. The gradient signal is clean.

### What Is Machine Epsilon

Machine epsilon is the smallest number that, when added to 1.0, produces a result different from 1.0 in floating-point arithmetic. For float64 (double precision), machine epsilon is approximately 2.2 x 10^-16.

When we compute a derivative via autograd and compare it to the known analytical answer, the difference should be on the order of machine epsilon (roughly 10^-15 to 10^-16). This is not an approximation error. It is the fundamental limit of representing real numbers in 64 bits. No numerical method can do better.

For float32 (single precision), machine epsilon is about 1.2 x 10^-7, so errors would be around 10^-7. This project uses float64 specifically to demonstrate that autograd achieves the theoretical best possible accuracy.

---

## Key Terminology

| Term | Definition |
|------|-----------|
| **Autograd** | PyTorch's automatic differentiation engine. Records operations on tensors and computes exact gradients via reverse-mode differentiation (backpropagation). |
| **Computational graph** | A directed acyclic graph that records the sequence of operations performed on tensors. Each node is an operation; edges represent data flow. Autograd traverses this graph backward to compute gradients. |
| **create_graph=True** | A flag passed to `torch.autograd.grad()` that tells PyTorch to also record the gradient computation itself in the graph. This allows computing higher-order derivatives (e.g., second derivatives) by differentiating through the first derivative. |
| **Finite differences** | A numerical method that approximates derivatives using function values at nearby points. Introduces truncation error proportional to the step size raised to some power. |
| **Symbolic differentiation** | Algebraic manipulation of mathematical expressions to produce derivative formulas. Exact but limited to closed-form expressions. |
| **Roundoff error** | The error introduced by representing real numbers in finite-precision floating point. Unavoidable in any numerical computation. For float64, roughly 10^-15 to 10^-16 relative to the value. |
| **float64 vs float32** | float64 (double precision) uses 64 bits to store a number, giving about 15-16 significant decimal digits. float32 (single precision) uses 32 bits, giving about 7 significant digits. This project uses float64 to clearly demonstrate machine-epsilon-level accuracy. |
| **Contour plot** | A 2D visualization where color represents the value of a function at each (x, t) point. Like a topographic map where color replaces elevation lines. Used here to show the function surface and error distributions. |

---

## What This Script Demonstrates

This script does NOT:
- Train a neural network
- Solve a differential equation
- Implement a PINN
- Use any optimizer or loss function

This script DOES:
- Compute derivatives of known functions using PyTorch autograd
- Compare those autograd derivatives to hand-derived analytical formulas
- Show that the errors are at machine epsilon (10^-14 to 10^-15)
- Verify that plugging an exact PDE solution into the autograd differentiation pipeline produces zero residual (to machine precision)

The purpose is to prove that the differentiation tool works before trusting it inside a PINN. It is the equivalent of calibrating a measuring instrument before using it for experiments.

---

## Part 1: Derivative Verification

### The Test Function

The test function is:

```
f(x, t) = sin(pi * x) * exp(-t)
```

defined on the domain x in [0, 1] and t in [0, 1], evaluated on a 31 x 31 grid.

Its analytical partial derivatives are:

```
df/dx   = pi * cos(pi * x) * exp(-t)
d2f/dx2 = -pi^2 * sin(pi * x) * exp(-t)
df/dt   = -sin(pi * x) * exp(-t)
```

These formulas are straightforward to derive by hand and provide a ground truth to compare against.

### How Autograd Computes the Derivatives

The script creates tensors `x` and `t` with `requires_grad=True`, computes `f = sin(pi*x) * exp(-t)`, and then calls:

```python
# First derivative with respect to x
f_x = torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f),
                          create_graph=True)[0]

# Second derivative with respect to x (differentiate f_x again)
f_xx = torch.autograd.grad(f_x, x, grad_outputs=torch.ones_like(f_x),
                           create_graph=True)[0]

# First derivative with respect to t
f_t = torch.autograd.grad(f, t, grad_outputs=torch.ones_like(f),
                          create_graph=True)[0]
```

The `create_graph=True` flag is essential for computing second derivatives. It tells PyTorch to record the first differentiation as part of the graph so that we can differentiate through it again.

### Results: Errors at Machine Epsilon

The absolute difference between the autograd derivatives and the analytical formulas is on the order of 10^-15. For example:

- Maximum error in f_x: ~10^-15
- Maximum error in f_xx: ~10^-14 to 10^-15
- Maximum error in f_t: ~10^-15

These errors are not approximation errors. They are roundoff errors from finite-precision arithmetic. The derivatives are as exact as floating-point numbers allow.

---

## Part 2: Heat Equation Residual Verification

### The Heat Equation and Its Exact Solution

The one-dimensional heat equation is:

```
u_t = alpha * u_xx
```

where alpha = 0.01 is the thermal diffusivity. This equation describes how temperature evolves over time in a material.

The exact analytical solution used for verification is:

```
u(x, t) = sin(pi * x) * exp(-pi^2 * alpha * t)
```

defined on x in [0, 1] and t in [0, 0.5], evaluated on a 101 x 61 grid.

You can verify by hand that this function satisfies u_t = alpha * u_xx exactly:

```
u_t   = -pi^2 * alpha * sin(pi * x) * exp(-pi^2 * alpha * t)
u_xx  = -pi^2 * sin(pi * x) * exp(-pi^2 * alpha * t)
alpha * u_xx = -pi^2 * alpha * sin(pi * x) * exp(-pi^2 * alpha * t)
```

So u_t - alpha * u_xx = 0 exactly.

### Computing Derivatives via Autograd

The script computes u from the exact formula using autograd-enabled tensors, then uses `torch.autograd.grad()` to obtain u_t and u_xx. This is the same differentiation pipeline that a PINN would use on a neural network output.

```python
u = torch.sin(pi * x) * torch.exp(-pi**2 * alpha * t)

u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                          create_graph=True)[0]

u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                          create_graph=True)[0]

u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x))[0]
```

### Residual Should Be Zero

The PDE residual is defined as:

```
residual = u_t - alpha * u_xx
```

Since the exact solution satisfies the heat equation perfectly, this residual is mathematically zero everywhere. The only nonzero values we observe come from floating-point roundoff.

### Results: Residual at Machine Epsilon

The computed residual has magnitude on the order of 10^-15 across the entire domain. This proves that:

1. Autograd computes u_t correctly.
2. Autograd computes u_xx correctly (via two successive differentiations).
3. The PINN differentiation pipeline (function -> autograd -> residual) produces exact results.

If a PINN using this pipeline shows nonzero residual, the error comes from the network not having learned the solution, not from the differentiation being approximate.

---

## Why This Matters for PINNs

This verification establishes a critical fact: autograd introduces zero approximation error into the PINN training loop. Consider the alternatives:

**If PINNs used finite differences for derivatives:**
- With grid spacing h = 0.01: approximation error ~ 10^-4
- With grid spacing h = 0.001: approximation error ~ 10^-6
- The network would need to be more accurate than the derivative approximation, which is extremely difficult to achieve and verify.
- The loss function would have a nonzero floor even for a perfect solution.

**With autograd:**
- Approximation error: exactly zero
- Roundoff error: ~10^-15 (irrelevant for training)
- The loss function reaches zero if and only if the network truly satisfies the PDE.
- Gradient signals to the optimizer are clean and meaningful.

This is why PINNs work. Automatic differentiation provides an exact bridge between the neural network output and the physics constraints. This project proves that bridge is trustworthy.

---

## Results

<img width="2969" height="877" alt="image" src="https://github.com/user-attachments/assets/6ed3ea3f-ef72-4023-b2c8-3429403e5043" />


The output is a three-panel figure:

**Left panel: Function contour**
Shows the test function f(x,t) = sin(pi*x)*exp(-t) as a filled contour plot over the (x, t) domain. This visualizes the smooth, well-behaved function being differentiated.

**Center panel: log10 error in f_xx**
Shows the base-10 logarithm of the absolute error in the second derivative d2f/dx2 computed by autograd versus the analytical formula. Values around -14 to -15 indicate errors at machine epsilon. The entire domain is uniformly at this level, confirming that autograd accuracy does not degrade anywhere.

**Right panel: log10 heat equation residual**
Shows the base-10 logarithm of the absolute residual |u_t - alpha*u_xx| for the exact heat equation solution. Values around -14 to -15 confirm that the residual is zero to machine precision. This is the definitive proof that the PINN differentiation pipeline works.

---

## Connection to Other Projects

This autograd verification is the foundation that all PINN projects build upon. The five PINN projects in this course each rely on the fact proven here:

1. **1D Heat Equation PINN** -- Uses autograd to compute u_t and u_xx of the network output. This verification proves those derivatives are exact.

2. **Burgers Equation PINN** -- Uses autograd for u_t, u_x, and u_xx. The nonlinear term u*u_x also gets exact derivatives because autograd handles products via the chain rule.

3. **Navier-Stokes PINN** -- Computes pressure gradients, velocity Laplacians, and continuity residuals via autograd. Multiple coupled derivatives all benefit from the exactness proven here.

4. **Schrodinger Equation PINN** -- Complex-valued derivatives (real and imaginary parts) computed via autograd. Same principle: chain rule applied exactly.

5. **Allen-Cahn Equation PINN** -- Nonlinear reaction term with cubic nonlinearity. Autograd differentiates through the nonlinearity exactly.

In every case, the PINN loss function includes terms of the form (PDE residual)^2. This project proves that when the network output is an exact solution, that residual is zero. Any nonzero residual during training is the network's fault, not the differentiation's fault.

---

## Requirements

- Python 3.8 or later
- PyTorch (any recent version with autograd support)
- NumPy
- Matplotlib

Install with:

```bash
pip install torch numpy matplotlib
```

---

## How to Run

```bash
cd Autograd_Verification
python autograd_verification.py
```

The script will:
1. Compute derivatives of the test function via autograd and compare to analytical formulas.
2. Print maximum absolute errors (expected: ~10^-14 to 10^-15).
3. Compute the heat equation residual for the exact solution.
4. Print maximum residual magnitude (expected: ~10^-14 to 10^-15).
5. Generate and save the three-panel verification plot.

No GPU is needed. The computation runs in under a second on any machine. Float64 precision is used throughout for clear roundoff assessment.
