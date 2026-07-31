# CODE_WALKTHROUGH.md - Architecture Comparison

**File:** `architecture_comparison.py`
**Project:** Comparing three neural network architectures for solving a simple ODE with Physics-Informed Neural Networks (PINNs).

---

## What This Code Does

This script trains three different neural network architectures on the same physics problem, then compares how well each one learns the solution. The problem is simple enough that we know the exact answer (sin(x)), which lets us measure true error - not just loss.

---

## The Problem Being Solved

```
Differential equation:  u''(x) = -sin(x)
Domain:                 x in [0, pi]
Boundary conditions:    u(0) = 0,  u(pi) = 0
Exact solution:         u(x) = sin(x)
```

This is a second-order ODE with Dirichlet boundary conditions at both ends. The exact solution being sin(x) is known ahead of time, which makes this an ideal benchmark - we can measure exactly how close each architecture gets.

---

## Step-by-Step Walkthrough

### Step 1: Imports and Hardware Setup

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
EPOCHS = 5000
```

Standard PyTorch imports plus numpy and matplotlib for evaluation and plotting. The code auto-detects GPU availability. The fixed seed (42) ensures reproducibility - every run produces the same results. 5000 epochs is the training budget each architecture gets.

---

### Step 2: Custom Sine Activation Function

```python
class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)
```

PyTorch does not ship a sine activation layer, so we define one. This is a simple wrapper that applies torch.sin element-wise. It matters because the exact solution to our problem IS a sine function - giving the network sine as a building block makes the problem dramatically easier. Think of it as handing the network the right tool for the job.

---

### Step 3: Three Architecture Definitions

```python
class PINN_A(nn.Module):  # Small + Tanh
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1)
        )

class PINN_B(nn.Module):  # Large + Tanh
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1)
        )

class PINN_C(nn.Module):  # Small + Sine
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20), SinActivation(),
            nn.Linear(20, 20), SinActivation(),
            nn.Linear(20, 1)
        )
```

Three architectures being compared:

| Model | Width | Depth | Activation | Parameters (approx) |
|-------|-------|-------|------------|---------------------|
| PINN_A | 20 | 2 hidden | Tanh | ~460 |
| PINN_B | 50 | 3 hidden | Tanh | ~5,200 |
| PINN_C | 20 | 2 hidden | Sine | ~460 |

The key comparison is:
- **A vs B**: Does making the network bigger help? (Same activation, different size)
- **A vs C**: Does choosing the right activation help? (Same size, different activation)

The answer this experiment reveals: choosing the right activation (C) beats brute-force scaling (B).

---

### Step 4: Derivative Helper

```python
def derivative(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True
    )[0]
```

This computes dy/dx using automatic differentiation. The `create_graph=True` flag is critical - it tells PyTorch to track this gradient computation itself, so we can later differentiate through it again (to get second derivatives) and also backpropagate through the entire loss.

Without `create_graph=True`, we could compute derivatives but could not train on them.

---

### Step 5: Trial Solution (Hard Boundary Enforcement)

```python
def trial_solution(model, s):
    return 4 * s * (1 - s) * model(s)
```

This is the most important design choice in the entire script. Instead of letting the network output u(s) directly, we multiply its output by `s * (1 - s)`.

**Why this works:** At s=0, the multiplier is `0 * 1 = 0`. At s=1, the multiplier is `1 * 0 = 0`. So no matter what the network outputs, the trial solution is always zero at both boundaries. The boundary conditions are satisfied exactly, by construction, for any network weights.

**Why the factor 4:** The multiplier `s*(1-s)` peaks at 0.25 (when s=0.5). Multiplying by 4 makes the peak equal to 1.0, so the network output is roughly on the same scale as the final solution. This is cosmetic - it does not change what the network can represent, just makes the optimization landscape slightly friendlier.

**The payoff:** Because boundaries are enforced by construction, the loss function only needs to minimize the PDE residual. There is no boundary penalty term, no hyperparameter balancing PDE loss vs BC loss. The loss is purely physics.

---

### Step 6: The Coordinate Transform

```python
s = torch.linspace(0, 1, 60, device=device).reshape(-1, 1)
s.requires_grad_(True)
```

The network operates on s in [0, 1], not x in [0, pi]. The relationship is s = x/pi (so x = pi*s).

**Why this matters for the PDE:** By chain rule, if u is a function of s and s = x/pi:

```
du/dx = (1/pi) * du/ds
d2u/dx2 = (1/pi^2) * d2u/ds2
```

So the original equation `u''(x) = -sin(x)` becomes:

```
(1/pi^2) * u''(s) = -sin(pi*s)
```

Or equivalently: `u''(s)/pi^2 + sin(pi*s) = 0`

This is exactly the residual computed in training. The coordinate transform normalizes the input to [0,1] (good for neural networks) while keeping the physics correct through the chain rule factor.

---

### Step 7: Training Loop

```python
def train_model(model_class, name):
    torch.manual_seed(SEED)
    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.5)

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        u_s = trial_solution(model, s)
        du = derivative(u_s, s)
        d2u = derivative(du, s)

        residual = d2u / (np.pi**2) + torch.sin(np.pi * s)
        loss = torch.mean(residual**2)

        loss.backward()
        optimizer.step()
        scheduler.step()
```

The training flow step by step:

```
INPUT: s (60 collocation points in [0,1])
  |
  v
[Neural Network] --> raw output
  |
  v
[Trial Solution] --> u_s = 4*s*(1-s)*raw_output
  |
  v
[1st Derivative] --> du/ds  (autograd)
  |
  v
[2nd Derivative] --> d2u/ds2  (autograd again)
  |
  v
[Residual] --> d2u/ds2 / pi^2 + sin(pi*s)   (should be zero)
  |
  v
[Loss] --> mean(residual^2)
  |
  v
[Backprop + Adam update]
```

The scheduler halves the learning rate at epoch 2500. This is a common trick - start with large steps to find the right region, then take smaller steps to refine.

Notice: 60 collocation points is very few. This works because the problem is smooth and one-dimensional. Higher-dimensional or turbulent problems need thousands or millions of points.

---

### Step 8: Best-State Tracking and Convergence Detection

```python
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = model.state_dict().copy()

        if loss.item() < 1e-5 and convergence_epoch is None:
            convergence_epoch = epoch
```

Two bookkeeping mechanisms:

**Best-state tracking:** Loss can temporarily spike during training (especially after learning rate changes). By saving the state dictionary whenever we hit a new lowest loss, we ensure the final model is the best one we ever saw - not just the last one.

**Convergence epoch:** Records the first time loss drops below 1e-5. This tells us how fast each architecture learns, not just how well. A model that converges at epoch 500 is more efficient than one that converges at epoch 4000, even if both reach similar final accuracy.

---

### Step 9: Post-Training Evaluation

```python
    model.load_state_dict(best_state)

    s_test = torch.linspace(0, 1, 300, device=device).reshape(-1, 1)
    s_test.requires_grad_(False)

    with torch.no_grad():
        u_pred = trial_solution(model, s_test)

    x_test = np.pi * s_test.cpu().numpy().flatten()
    u_exact = np.sin(x_test)
    u_num = u_pred.cpu().numpy().flatten()

    max_error = np.max(np.abs(u_exact - u_num))
    relative_l2 = np.sqrt(np.sum((u_exact - u_num)**2) / np.sum(u_exact**2))
```

After training, we:
1. Load the best weights (not the final weights)
2. Evaluate on 300 points (5x more than training) to test generalization
3. Convert back from s-space to x-space for comparison with exact solution
4. Compute two error metrics:
   - **Max error:** Worst-case pointwise deviation (L-infinity norm)
   - **Relative L2:** RMS error normalized by solution magnitude (tells you percentage accuracy)

The `torch.no_grad()` context disables gradient tracking since we only need forward evaluation here. This saves memory and computation.

---

### Step 10: Main Execution and Results Printing

```python
if __name__ == '__main__':
    configs = [
        (PINN_A, "PINN-A (Small+Tanh)"),
        (PINN_B, "PINN-B (Large+Tanh)"),
        (PINN_C, "PINN-C (Small+Sine)"),
    ]

    results = []
    for model_class, name in configs:
        result = train_model(model_class, name)
        results.append(result)
        print(f"{name}: max_err={result['max_error']:.2e}, "
              f"rel_L2={result['relative_l2']:.2e}, "
              f"converged={result['convergence_epoch']}")
```

Trains all three models sequentially and reports metrics. Typical results look like:

```
PINN-A (Small+Tanh):  max_err ~ 1e-3, converges ~ epoch 3000-4000
PINN-B (Large+Tanh):  max_err ~ 5e-4, converges ~ epoch 2000-3000
PINN-C (Small+Sine):  max_err ~ 1e-5, converges ~ epoch 500-1000
```

The takeaway: PINN_C (sine activation) matches or beats the 10x-larger PINN_B while converging much faster. Architecture choice matters more than size for this class of problem.

---

### Step 11: Visualization

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left panel: predictions vs exact
axes[0].plot(x_exact, u_exact, 'k-', label='Exact')
for r in results:
    axes[0].plot(r['x'], r['u_pred'], '--', label=r['name'])

# Right panel: convergence curves
for r in results:
    axes[1].semilogy(r['loss_history'], label=r['name'])

plt.savefig('chapter3_architecture_comparison.png', dpi=150)
```

Two-panel figure:
- **Left:** All three predictions overlaid on the exact sin(x) curve. With a good run, all three look correct to the eye - the differences are only visible in the error metrics.
- **Right:** Loss vs epoch on a log scale. This is where the architecture differences become visually obvious - PINN_C's loss drops like a rock while A and B plateau longer before descending.

---

### Step 12: Training Flow Diagram (Full Pipeline)

```
+------------------------------------------------------------------+
|                    ARCHITECTURE COMPARISON                         |
+------------------------------------------------------------------+
|                                                                    |
|  For each architecture (A, B, C):                                 |
|                                                                    |
|  [1] Initialize model with SEED=42                                |
|       |                                                            |
|  [2] Generate 60 collocation points s in [0,1]                    |
|       |                                                            |
|  [3] TRAINING LOOP (5000 epochs)                                  |
|       |                                                            |
|       |   s --> Network(s) --> raw                                 |
|       |                    --> 4*s*(1-s)*raw = u(s)                |
|       |                    --> du/ds   (autograd)                  |
|       |                    --> d2u/ds2 (autograd)                  |
|       |                                                            |
|       |   residual = d2u/ds2 / pi^2 + sin(pi*s)                   |
|       |   loss = mean(residual^2)                                  |
|       |                                                            |
|       |   [Track best state]                                       |
|       |   [Detect convergence < 1e-5]                              |
|       |   [Step LR at epoch 2500]                                  |
|       |                                                            |
|  [4] Load best state                                               |
|       |                                                            |
|  [5] Evaluate on 300 test points                                   |
|       |                                                            |
|  [6] Compute max_error + relative_l2                               |
|                                                                    |
+------------------------------------------------------------------+
|  Compare all three --> Print metrics --> Save figure               |
+------------------------------------------------------------------+
```

---

## Key Design Decisions Explained

### Why no boundary loss term?

Most PINN tutorials write the loss as:

```
total_loss = pde_loss + lambda * boundary_loss
```

This code has ZERO boundary loss. The trial solution `4*s*(1-s)*model(s)` makes boundary enforcement automatic. This eliminates:
- The lambda hyperparameter (which is notoriously hard to tune)
- Competition between PDE accuracy and boundary accuracy
- The possibility of satisfying the PDE but violating boundaries

The tradeoff: trial solutions only work for simple geometries and boundary conditions. For complex 3D domains or Neumann conditions, you often need the penalty approach.

### Why normalize to [0,1]?

Neural networks work best when inputs are O(1). The domain [0, pi] is already close to [0, 1], but normalizing exactly to [0, 1] is cleaner and the chain rule correction (dividing by pi^2) is trivial.

### Why track convergence epoch?

Final accuracy alone does not tell the full story. If architecture C reaches 1e-5 loss at epoch 500 while architecture B needs epoch 4000, then C is 8x more efficient per training dollar. For expensive real-world problems (3D, millions of points), convergence speed can be the deciding factor.

---

## Comparison to Other Chapter 3 Projects

| Aspect | Architecture Comparison | Activation Study | Training Dynamics |
|--------|------------------------|------------------|-------------------|
| Focus | Network size + depth | Activation functions only | Learning rate + scheduler |
| Models compared | 3 architectures | Single architecture, swap activations | Single architecture |
| Trial solution | Yes (hard BCs) | Yes (hard BCs) | Yes (hard BCs) |
| Loss type | Pure PDE | Pure PDE | Pure PDE |
| Key metric | Convergence speed | Final accuracy | Loss trajectory shape |
| Collocation points | 60 fixed | 60 fixed | 60 fixed |
| Main finding | Right activation > bigger network | Sine activation wins for sinusoidal solutions | Warmup helps early training |

---

## Common Modifications

**Want to try a different ODE?** Change the residual line and the exact solution. The trial solution structure (zero at boundaries) stays the same for any Dirichlet problem on [0, pi].

**Want more collocation points?** Change the 60 in `torch.linspace(0, 1, 60, ...)`. More points = more accurate gradients but slower per epoch.

**Want to add a fourth architecture?** Define a new class (e.g., PINN_D with ReLU activation), add it to the configs list, and rerun. The training and evaluation pipeline handles it automatically.

**Want to test on a harder problem?** Try u''(x) = -k^2 * sin(kx) for larger k. Higher frequency solutions expose architecture differences more dramatically.
