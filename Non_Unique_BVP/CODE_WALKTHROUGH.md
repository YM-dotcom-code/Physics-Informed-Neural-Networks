# CODE_WALKTHROUGH.md - Trivial Solution Demo

## Project: Physics-Informed Neural Network for Non-Unique PDE Solutions

**File:** `trivial_solution_demo.py`
**Purpose:** Demonstrate how PINNs can fail when a PDE has infinitely many solutions, and how a single amplitude constraint rescues the problem.

---

## Table of Contents

1. [The Core Problem](#1-the-core-problem)
2. [Imports and Device Setup](#2-imports-and-device-setup)
3. [Problem Statement in Code](#3-problem-statement-in-code)
4. [PINN Architecture](#4-pinn-architecture)
5. [Derivative Helper](#5-derivative-helper)
6. [Trial Solution Design](#6-trial-solution-design)
7. [Training Function Structure](#7-training-function-structure)
8. [Residual Computation](#8-residual-computation)
9. [Best-State Tracking and Evaluation](#9-best-state-tracking-and-evaluation)
10. [Main Execution and Case Comparison](#10-main-execution-and-case-comparison)
11. [Visualization: Three-Panel Plot](#11-visualization-three-panel-plot)
12. [Key Takeaways and Cross-Project Comparison](#12-key-takeaways-and-cross-project-comparison)

---

## 1. The Core Problem

### The PDE

```
u''(x) + u(x) = 0,   x in [0, pi]
u(0) = 0,  u(pi) = 0
```

### Why This PDE is Special

The characteristic equation for `u'' + u = 0` is `r^2 + 1 = 0`, giving roots `r = +/- i`. The general solution is:

```
u(x) = A*cos(x) + B*sin(x)
```

Applying the boundary conditions:
- `u(0) = 0` forces `A = 0`
- `u(pi) = B*sin(pi) = B*0 = 0` is satisfied for ANY value of B

So the solution is `u(x) = C*sin(x)` where C is completely undetermined. This is an eigenvalue problem: lambda=1 is an eigenvalue of the negative-second-derivative operator with these boundary conditions, and `sin(x)` is the corresponding eigenfunction.

### The Neural Network Trap

Neural networks are typically initialized with small random weights, producing outputs near zero at the start of training. When we train a PINN on this problem:

1. The network starts near the zero function
2. The zero function `u(x) = 0` IS a valid solution (C=0)
3. The loss is already near zero from initialization
4. Gradient descent has no reason to move away from zero

The network converges to the trivial solution not because it is "wrong" but because the problem is genuinely underdetermined. The PINN has no way to pick a nonzero amplitude without additional information.

---

## 2. Imports and Device Setup

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
```

**What happens here:**
- Standard PyTorch and visualization imports
- Device selection for GPU acceleration when available
- Global seed for reproducibility across both training cases

---

## 3. Problem Statement in Code

```python
# PDE: u''(x) + u(x) = 0, x in [0, pi]
# BCs: u(0) = 0, u(pi) = 0
# General solution: u(x) = C*sin(x), amplitude C is undetermined
```

This comment block documents the mathematical setup. The key word is "undetermined." Unlike most PINN demos where the PDE plus boundary conditions yield a unique solution, this problem has a one-parameter family of solutions.

---

## 4. PINN Architecture

```python
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.net(x)
```

**Architecture breakdown:**

```
Input(1) --> Linear(1,20) --> Tanh --> Linear(20,20) --> Tanh --> Linear(20,20) --> Tanh --> Linear(20,1) --> Output(1)
```

- **3 hidden layers**, each with 20 neurons
- **Tanh activation** throughout (smooth, infinitely differentiable, needed for second derivatives)
- **Layer widths:** [1, 20, 20, 20, 1]

This is the same architecture used across projects in this series. The network is deliberately standard so the lesson is about the *problem formulation*, not the network design.

---

## 5. Derivative Helper

```python
def derivative(y, x, order=1):
    for _ in range(order):
        y = torch.autograd.grad(
            y, x,
            grad_outputs=torch.ones_like(y),
            create_graph=True
        )[0]
    return y
```

**How it works:**
- Uses PyTorch autograd to compute dy/dx through the computational graph
- `grad_outputs=torch.ones_like(y)` handles batched (vector-valued) differentiation
- `create_graph=True` allows higher-order derivatives by keeping the graph alive
- The loop applies differentiation repeatedly for `order > 1`

This is the same autograd pattern used in `architecture_comparison.py` and other projects. It computes exact symbolic derivatives through the network, not finite differences.

---

## 6. Trial Solution Design

This is the intellectual heart of the demo. Two modes exist:

```python
def trial_solution(model, s, enforce_amplitude):
    if not enforce_amplitude:
        # Only enforces BCs: u(0) = u(pi) = 0
        return 4 * s * (1 - s) * model(s)
    else:
        # Enforces BCs AND u(pi/2) = 1
        return 4 * s * (1 - s) * (1 + (s - 0.5) * model(s))
```

Here `s` is the normalized coordinate: `s = x / pi`, so `s in [0, 1]`.

### Mode 1: Boundary Conditions Only (`enforce_amplitude=False`)

```
u_trial(s) = 4 * s * (1-s) * NN(s)
```

- At `s=0`: `4 * 0 * 1 * NN(0) = 0` (left BC satisfied)
- At `s=1`: `4 * 1 * 0 * NN(1) = 0` (right BC satisfied)
- The factor `4*s*(1-s)` is a "lifting function" that zeros out at both boundaries
- The network output is unconstrained in between

**Problem:** The network can output zero everywhere and satisfy the PDE perfectly. It will.

### Mode 2: Boundary Conditions Plus Amplitude (`enforce_amplitude=True`)

```
u_trial(s) = 4 * s * (1-s) * (1 + (s-0.5) * NN(s))
```

Let us verify what happens at the three key points:

**At s=0 (left boundary):**
```
4 * 0 * (1-0) * (...) = 0    [BC satisfied regardless of NN]
```

**At s=1 (right boundary):**
```
4 * 1 * (1-1) * (...) = 0    [BC satisfied regardless of NN]
```

**At s=0.5 (midpoint, corresponds to x=pi/2):**
```
4 * 0.5 * 0.5 * (1 + (0.5-0.5)*NN(0.5))
= 4 * 0.25 * (1 + 0*NN(0.5))
= 1 * (1)
= 1
```

The `(s-0.5)` factor kills the network contribution at the midpoint. The remaining constant structure gives exactly `4 * 0.25 * 1 = 1`. So `u(pi/2) = 1` is hardwired.

**Critical insight:** This does NOT prescribe the shape of the solution. It only fixes the amplitude. The network must still learn from the PDE that the correct shape is sinusoidal. If you set `u(pi/2) = 1`, then `C = 1/sin(pi/2) = 1`, giving the unique solution `u(x) = sin(x)`.

---

## 7. Training Function Structure

```python
def train_case(enforce_amplitude, epochs=5000):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.5)

    s = torch.linspace(0, 1, 60, device=device).reshape(-1, 1)
    s.requires_grad_(True)

    # ... training loop ...
```

**Design choices:**
- **Seeds reset** before each case for fair comparison
- **Adam optimizer** at lr=1e-3 (standard for PINNs)
- **StepLR scheduler:** halves the learning rate at epoch 2500 for fine-tuning
- **60 collocation points:** evenly spaced in [0,1], enough for this smooth problem
- **requires_grad_(True):** enables autograd to differentiate through `s`

---

## 8. Residual Computation

```python
for epoch in range(epochs):
    optimizer.zero_grad()

    u = trial_solution(model, s, enforce_amplitude)
    u_s = derivative(u, s, order=1)
    u_ss = derivative(u, s, order=2)

    # PDE: u''(x) + u(x) = 0
    # In normalized coords: (1/pi^2) * u_ss + u = 0
    residual = u_ss / (np.pi ** 2) + u
    loss = torch.mean(residual ** 2)

    loss.backward()
    optimizer.step()
    scheduler.step()
```

### Why `u_ss / pi^2`?

The coordinate transformation `s = x/pi` means:
```
du/dx = (du/ds) * (ds/dx) = (1/pi) * du/ds
d2u/dx2 = (1/pi^2) * d2u/ds2
```

So the PDE `u''(x) + u(x) = 0` in normalized coordinates becomes:
```
(1/pi^2) * u_ss + u = 0
```

### Why the residual uses `u` (not `sin(pi*s)`)

This is a homogeneous PDE. There is no forcing term. The residual is simply:
```
residual = u'' + u
```

Compare with a Poisson-type problem like `u'' = -sin(pi*x)` where the right-hand side would appear in the residual. Here, the "source" is zero, and the PDE constraint alone (with BCs) yields a family of solutions.

---

## 9. Best-State Tracking and Evaluation

```python
    best_loss = float('inf')
    best_state = None
    losses = []

    for epoch in range(epochs):
        # ... compute loss ...
        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best model
    model.load_state_dict(best_state)
```

After training, evaluation on a finer grid:

```python
    # Evaluate on 300 points
    s_eval = torch.linspace(0, 1, 300, device=device).reshape(-1, 1)
    s_eval.requires_grad_(True)

    u_pred = trial_solution(model, s_eval, enforce_amplitude)
    u_pred_np = u_pred.detach().cpu().numpy().flatten()

    x_eval = s_eval.detach().cpu().numpy().flatten() * np.pi
    max_amplitude = np.max(np.abs(u_pred_np))

    # Compare against sin(x) (the C=1 solution)
    u_exact = np.sin(x_eval)
    max_error = np.max(np.abs(u_pred_np - u_exact))
    relative_l2 = np.sqrt(np.sum((u_pred_np - u_exact)**2) / np.sum(u_exact**2))
```

The function returns a dictionary with all metrics and arrays for plotting.

---

## 10. Main Execution and Case Comparison

```python
if __name__ == '__main__':
    print("Case 1: Underdetermined (BCs only)")
    result1 = train_case(enforce_amplitude=False)

    print("Case 2: Well-posed (BCs + amplitude constraint)")
    result2 = train_case(enforce_amplitude=True)

    # Print comparison
    print(f"Case 1 max amplitude: {result1['max_amplitude']:.6f}")
    print(f"Case 2 max amplitude: {result2['max_amplitude']:.6f}")
    print(f"Case 2 relative L2 error: {result2['relative_l2']:.6e}")
```

**Expected results:**
- Case 1: max amplitude near 0 (trivial solution found)
- Case 2: max amplitude near 1 (correct `sin(x)` recovered)
- Case 2 relative L2 error: very small (order 1e-4 or better)

---

## 11. Visualization: Three-Panel Plot

The script produces a three-panel figure:

```
+-------------------+-------------------+-------------------+
|                   |                   |                   |
|   Loss History    |   Underdetermined |    Constrained    |
|   (both cases)    |   (near zero)     |    (matches sin)  |
|                   |                   |                   |
+-------------------+-------------------+-------------------+
   Panel 1              Panel 2              Panel 3
```

**Panel 1 - Convergence:**
- Log-scale loss vs. epoch for both cases
- Case 1 drops quickly (zero is easy to find)
- Case 2 takes longer but converges to a meaningful solution

**Panel 2 - Underdetermined prediction:**
- Shows the Case 1 output (near-zero flat line)
- Overlays the exact `sin(x)` for reference
- Visually demonstrates the trivial solution trap

**Panel 3 - Constrained prediction:**
- Shows the Case 2 output matching `sin(x)` closely
- Demonstrates that one extra constraint resolves non-uniqueness

---

## 12. Key Takeaways and Cross-Project Comparison

### Training Flow Diagram

```
                    +-------------------+
                    |   Define PDE:     |
                    |   u'' + u = 0     |
                    +--------+----------+
                             |
                    +--------v----------+
                    | Choose trial form |
                    +--------+----------+
                             |
              +--------------+--------------+
              |                             |
   +----------v----------+     +-----------v-----------+
   | enforce_amplitude=F |     | enforce_amplitude=T   |
   | u = 4s(1-s)*NN(s)  |     | u = 4s(1-s)*(1+      |
   |                     |     |     (s-0.5)*NN(s))    |
   +----------+----------+     +-----------+-----------+
              |                             |
   +----------v----------+     +-----------v-----------+
   | Compute residual:   |     | Compute residual:     |
   | u_ss/pi^2 + u       |     | u_ss/pi^2 + u        |
   +----------+----------+     +-----------+-----------+
              |                             |
   +----------v----------+     +-----------v-----------+
   | Loss -> 0 quickly   |     | Loss -> 0 gradually   |
   | (zero IS a solution)|     | (learns sine shape)   |
   +----------+----------+     +-----------+-----------+
              |                             |
   +----------v----------+     +-----------v-----------+
   | Output: u ~ 0       |     | Output: u ~ sin(x)   |
   | (trivial, valid)    |     | (correct, unique)     |
   +---------------------+     +-----------------------+
```

### Comparison to Other Projects

| Aspect | trivial_solution_demo.py | architecture_comparison.py |
|--------|--------------------------|----------------------------|
| PDE type | Homogeneous (u''+u=0) | Poisson (u''=-sin) |
| Solution uniqueness | Non-unique (family) | Unique |
| Main lesson | Problem formulation | Network design choices |
| Trial solution | 2 variants tested | 1 standard form |
| Failure mode shown | Trivial solution trap | Poor approximation |
| Fix demonstrated | Amplitude constraint | Better architecture |
| Number of cases | 2 | Multiple architectures |
| Collocation points | 60 | Varies |
| Hidden layers | 3 (fixed) | Varies (the point) |

### Why This Matters for Practitioners

1. **Low loss does not mean correct solution.** Case 1 achieves near-zero loss but finds the trivial solution. Always validate against known behavior.

2. **Ill-posed problems need extra constraints.** PINNs inherit the mathematical properties of the underlying PDE. If the continuous problem is underdetermined, the discrete (neural network) problem will be too.

3. **Trial solutions can encode more than BCs.** The `enforce_amplitude=True` form shows that algebraic design of the trial function can impose interior constraints without adding loss terms.

4. **Neural network initialization bias is real.** Small initial weights plus a valid zero solution create a basin of attraction that gradient descent will not escape.

---

## Summary

This demo isolates a failure mode that is distinct from network capacity or training issues: the PINN "fails" because the mathematical problem itself does not have a unique solution. The fix is not a better optimizer or deeper network. It is a better problem formulation. Adding one scalar constraint (the amplitude at the midpoint) transforms the problem from underdetermined to well-posed, and the same network with the same optimizer immediately recovers the correct solution.
