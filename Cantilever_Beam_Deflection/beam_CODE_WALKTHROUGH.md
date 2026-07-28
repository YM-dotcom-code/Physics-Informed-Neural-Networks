# CODE WALKTHROUGH: Cantilever Beam PINN

## Physics-Informed Neural Network for Euler-Bernoulli Beam Deflection

This walkthrough dissects every section of the cantilever beam PINN code, explaining the physics, the numerical strategy, and each implementation choice.

---

## Training Flow

```
+------------------+     +-------------------+     +--------------------+
| Collocation Pts  |     | Neural Network    |     | Automatic          |
| 60 pts in [0,1]  |---->| 3x30 tanh layers  |---->| Differentiation    |
| requires_grad    |     | s^2 * net(s)      |     | v -> v' -> v'' ... |
+------------------+     +-------------------+     +--------------------+
                                                            |
                    +---------------------------------------+
                    |
                    v
+------------------------------------------+
| Loss Computation                         |
|                                          |
|  PDE:  mean( (v'''' - 1)^2 )            |
|  BC:   10 * mean( v''(1)^2 + v'''(1)^2 )|
|                                          |
|  Total = loss_pde + loss_free            |
+------------------------------------------+
                    |
                    v
+------------------------------------------+
| Adam Optimizer + StepLR Scheduler        |
| lr=1e-3, step_size=4000, gamma=0.5      |
| Best-state tracking over 8000 epochs    |
+------------------------------------------+

```

---

## Section 1: Imports and Device Setup

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

```

**What this does:**

- `torch` and `torch.nn` provide the neural network framework and autograd engine.
- `numpy` handles post-processing arrays for evaluation and plotting.
- `matplotlib` generates the final comparison plots.
- Seeds are fixed at 42 for full reproducibility across runs.
- CUDA is selected automatically if a GPU is available, otherwise CPU is used.

**Why these choices:**

For a 4th-order PDE, automatic differentiation through a computational graph is essential. PyTorch's `autograd.grad` with `create_graph=True` allows us to differentiate through derivatives themselves, building up to the 4th derivative needed for the beam equation.

---

## Section 2: Physical Parameters

```python
E = 200e9       # Young's modulus [Pa]
I = 8.33e-6     # Second moment of area [m^4]
q = 1000.0      # Distributed load [N/m]
L = 1.0         # Beam length [m]
EI = E * I      # Flexural rigidity [N*m^2]

```

**Physical meaning:**

| Parameter | Value | Units | Description |
| --- | --- | --- | --- |
| E | 200e9 | Pa | Steel elastic modulus |
| I | 8.33e-6 | m^4 | Cross-section property |
| q | 1000 | N/m | Uniform load intensity |
| L | 1.0 | m | Span length |
| EI | 1.666e6 | N*m^2 | Bending stiffness (product) |

**The governing equation:**

The Euler-Bernoulli beam equation for a uniform beam under distributed load:

```
EI * w''''(x) = q

```

where w(x) is the transverse deflection. This is a 4th-order ODE requiring four boundary conditions.

**Boundary conditions for a cantilever (fixed-free):**

- Fixed end (x=0): w(0) = 0 (no displacement), w'(0) = 0 (no rotation)
- Free end (x=L): w''(L) = 0 (no bending moment), w'''(L) = 0 (no shear force)

---

## Section 3: Scaling and Nondimensionalization

```python
w_scale = q * L**4 / EI

```

**What this computes:**

`w_scale` is the characteristic deflection magnitude. For our parameters:

```
w_scale = 1000 * 1.0^4 / 1.666e6 = 6.0e-4 m

```

**The nondimensional transformation:**

We define:

- s = x / L (nondimensional coordinate, s in [0,1])
- v(s) = w(x) / w_scale (nondimensional deflection)

Substituting into the beam equation:

```
EI * w''''(x) = q

w = w_scale * v
x = L * s
d/dx = (1/L) * d/ds
d^4/dx^4 = (1/L^4) * d^4/ds^4

EI * (w_scale / L^4) * v''''(s) = q
EI * (q*L^4/EI) / L^4 * v''''(s) = q
q * v''''(s) = q

v''''(s) = 1

```

**Why nondimensionalize for a 4th-order problem:**

This is even more critical than for 2nd-order problems:

1. **Extreme scale separation.** The raw deflection is O(10^-4) meters while EI is O(10^6). The 4th derivative amplifies any scale mismatch by L^4. Without scaling, the network would need to output values near 0.0006 while the PDE residual involves multiplying by 1.666e6.
2. **Unit RHS.** The nondimensional equation v''''(s) = 1 has a right-hand side of unity. The network outputs and PDE residual live on the same O(1) scale, preventing gradient pathology.
3. **Domain normalization.** With L=1 already, the spatial coordinate s naturally falls in [0,1]. For general L, this mapping would also normalize the domain.
4. **Derivative magnification.** Each differentiation divides by L. For the 4th derivative, errors scale as 1/L^4. Keeping everything O(1) prevents catastrophic loss of precision in the higher derivatives.

---

## Section 4: Analytical Solution

```python
def exact_deflection(x):
    return q * (x**4 - 4*L*x**3 + 6*L**2*x**2) / (24 * EI)

```

**Derivation:**

Starting from v''''(s) = 1, integrate four times:

```
v'''(s) = s + C1
v''(s)  = s^2/2 + C1*s + C2
v'(s)   = s^3/6 + C1*s^2/2 + C2*s + C3
v(s)    = s^4/24 + C1*s^3/6 + C2*s^2/2 + C3*s + C4

```

Apply boundary conditions:

- v(0) = 0: C4 = 0
- v'(0) = 0: C3 = 0
- v''(1) = 0: 1/2 + C1 + C2 = 0
- v'''(1) = 0: 1 + C1 = 0, so C1 = -1

From v''(1) = 0: C2 = -1/2 - C1 = -1/2 + 1 = 1/2

So: v(s) = s^4/24 - s^3/6 + s^2/4

Converting back to physical units (x = s*L, w = w_scale * v):

```
w(x) = (q*L^4/EI) * [(x/L)^4/24 - (x/L)^3/6 + (x/L)^2/4]
     = q * [x^4 - 4*L*x^3 + 6*L^2*x^2] / (24*EI)

```

**Verification at tip (x=L):**

```
w(L) = q * (L^4 - 4*L^4 + 6*L^4) / (24*EI)
     = q * 3*L^4 / (24*EI)
     = q*L^4 / (8*EI)

```

This matches the classical formula for cantilever tip deflection under uniform load.

---

## Section 5: Derivative Helper

```python
def derivative(y, x):
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True
    )[0]

```

**What this does:**

Computes dy/dx using reverse-mode automatic differentiation. The key argument is `create_graph=True`, which ensures the derivative operation itself is recorded in the computational graph.

**Why **`create_graph=True`** is essential:**

For the beam equation we need the 4th derivative. Each call to `derivative()` must produce a result that can itself be differentiated:

```
v1 = derivative(v, s)       # v' = dv/ds
v2 = derivative(v1, s)      # v'' = d^2v/ds^2
v3 = derivative(v2, s)      # v''' = d^3v/ds^3
v4 = derivative(v3, s)      # v'''' = d^4v/ds^4

```

Without `create_graph=True`, v1 would be a leaf tensor with no graph connection, and `derivative(v1, s)` would fail or return zero.

**Why **`grad_outputs=torch.ones_like(y)`**:**

The `autograd.grad` function computes vector-Jacobian products. Since our output y has shape (N,1), we pass a ones vector to get the actual derivative values at each collocation point, not a summed scalar gradient.

**Computational cost note:**

Four nested differentiations create a deep computational graph. For tanh networks, each differentiation produces hyperbolic function compositions that grow in complexity. The 30-neuron width keeps this manageable while maintaining expressiveness for the smooth beam solution.

---

## Section 6: Neural Network Architecture

```python
class BeamNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, 1)
        )

    def forward(self, s):
        return s**2 * self.net(s)

```

**Architecture breakdown:**

```
Input s (1)
    |
Linear(1, 30) + Tanh
    |
Linear(30, 30) + Tanh
    |
Linear(30, 30) + Tanh
    |
Linear(30, 1)
    |
    v
raw = self.net(s)
    |
    v
output = s^2 * raw    <-- Hard constraint embedding

```

**Network dimensions:**

- Parameters in layer 1: 1*30 + 30 = 60
- Parameters in layer 2: 30*30 + 30 = 930
- Parameters in layer 3: 30*30 + 30 = 930
- Parameters in layer 4: 30*1 + 1 = 31
- Total: 1951 trainable parameters

**Why tanh activation:**

The beam deflection is a smooth C-infinity function (a 4th-degree polynomial). Tanh is infinitely differentiable, so all four derivatives exist and are smooth. ReLU would fail here because its 2nd derivative is zero everywhere (and 3rd/4th are undefined at the origin). Even for the 4th derivative, tanh provides well-behaved gradients.

**The s^2 hard constraint -- detailed proof:**

The output form `v(s) = s^2 * N(s)` where N(s) = self.net(s) enforces both fixed-end boundary conditions exactly, regardless of what the network learns.

**Proof that v(0) = 0:**

```
v(0) = 0^2 * N(0) = 0

```

Trivially satisfied for any network output N(0).

**Proof that v'(0) = 0 (chain rule):**

```
v(s) = s^2 * N(s)

v'(s) = d/ds [s^2 * N(s)]
      = 2*s * N(s) + s^2 * N'(s)    [product rule]

v'(0) = 2*0 * N(0) + 0^2 * N'(0)
      = 0 + 0
      = 0

```

Both terms vanish at s=0. The first term has the factor 2*s, the second has s^2. Neither depends on N(0) or N'(0). This is the key insight: the s^2 prefactor provides a double zero at s=0, enforcing both the function value and its first derivative to vanish there.

**Why not s^1 (just s*N(s))?**

```
v(s) = s * N(s)
v'(s) = N(s) + s * N'(s)
v'(0) = N(0)    <-- NOT guaranteed to be zero!

```

A linear prefactor only enforces v(0)=0, not v'(0)=0. We need s^2 specifically because the cantilever has two conditions at the fixed end.

**Why not s^3 or higher?**

Using s^3 would also enforce v''(0)=0, which is NOT a boundary condition for the cantilever. The bending moment at the fixed end is generally nonzero (it equals qL^2/2). Over-constraining would prevent the network from representing the true solution.

**Connection to the analytical solution:**

The exact nondimensional solution is v(s) = s^4/24 - s^3/6 + s^2/4. Factoring out s^2: v(s) = s^2 * (s^2/24 - s/6 + 1/4). The network learns to approximate the polynomial (s^2/24 - s/6 + 1/4), which equals 0.25 at s=0 and varies smoothly.

---

## Section 7: Optimizer, Scheduler, and Best-State Tracking

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4000, gamma=0.5)

best_loss = float('inf')
best_state = None

```

**Adam optimizer (lr=1e-3):**

Adam combines momentum with adaptive per-parameter learning rates. For PINNs with 4th-order derivatives, the loss landscape can have sharp curvature variations. Adam's adaptive step sizes help navigate regions where the 4th derivative creates steep gradients alongside flat plateaus.

**StepLR scheduler (step_size=4000, gamma=0.5):**

The learning rate schedule:

| Epochs | Learning Rate |
| --- | --- |
| 0-3999 | 1.0e-3 |
| 4000-7999 | 5.0e-4 |

The single halving at epoch 4000 divides training into two phases:

1. Exploration (epochs 0-3999): Larger steps find the basin of the solution.
2. Refinement (epochs 4000-7999): Smaller steps polish the approximation, especially important for the higher-order derivatives to converge.

**Best-state tracking:**

```python
best_loss = float('inf')
best_state = None

```

Training a 4th-order PINN can exhibit loss oscillations, particularly because the PDE residual involves v'''' which amplifies small network perturbations. Saving the best state ensures we recover the lowest-loss configuration even if later epochs introduce transient instabilities.

---

## Section 8: Collocation Points

```python
s = torch.linspace(0, 1, 60, device=device).reshape(-1, 1).requires_grad_(True)
sL = torch.ones(1, 1, device=device).requires_grad_(True)

```

**Interior collocation (60 points):**

60 uniformly spaced points in [0, 1] serve as the locations where the PDE residual v''''(s) - 1 = 0 is enforced. The `requires_grad_(True)` flag is essential because `torch.autograd.grad` needs to differentiate with respect to the input coordinates.

**Why 60 points:**

For a 4th-degree polynomial solution, even a small number of collocation points suffices theoretically. In practice, 60 points provides enough spatial resolution for the network to learn the 4th derivative accurately without excessive computational cost per epoch. The 4th derivative amplifies any spatial aliasing, so having adequate density matters more here than for lower-order equations.

**Boundary evaluation point (sL):**

```python
sL = torch.ones(1, 1, device=device).requires_grad_(True)

```

This is a single point at s=1 (the free end) used exclusively for evaluating the free-end boundary conditions v''(1)=0 and v'''(1)=0. It needs its own `requires_grad` flag because we differentiate v with respect to sL to compute v''(sL) and v'''(sL).

**Why not include sL in the main collocation array:**

Separating the boundary point makes the code cleaner and allows independent weighting of PDE vs BC losses. The fixed-end conditions at s=0 are already hard-coded via the s^2 prefactor, so only the free-end conditions need explicit enforcement.

**Why no boundary point at s=0:**

The s^2 architecture satisfies w(0)=0 and w'(0)=0 exactly. No loss term is needed for the fixed end. This is the advantage of hard constraint embedding -- it removes two conditions from the loss function entirely.

---

## Section 9: Training Loop

```python
for epoch in range(8000):
    optimizer.zero_grad()

    # Forward pass through collocation points
    v = model(s)

    # Compute derivatives up to 4th order
    v1 = derivative(v, s)
    v2 = derivative(v1, s)
    v3 = derivative(v2, s)
    v4 = derivative(v3, s)

    # PDE residual loss: v'''' = 1
    loss_pde = torch.mean((v4 - 1)**2)

    # Free-end boundary conditions at s=1
    vL = model(sL)
    v1L = derivative(vL, sL)
    v2L = derivative(v1L, sL)
    v3L = derivative(v2L, sL)

    loss_free = torch.mean(v2L**2 + v3L**2)

    # Total loss with BC weighting
    loss = loss_pde + 10 * loss_free

    loss.backward()
    optimizer.step()
    scheduler.step()

    # Track best model
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_state = model.state_dict().copy()

```

**Step-by-step breakdown of one epoch:**

**Step 1: Zero gradients**

`optimizer.zero_grad()` clears accumulated gradients from the previous iteration. Without this, gradients would accumulate across epochs.

**Step 2: Forward pass**

`v = model(s)` computes v(s) = s^2 * net(s) at all 60 collocation points. Output shape: (60, 1).

**Step 3: Four successive differentiations**

```
v1 = derivative(v, s)    # v'(s),   shape (60, 1)
v2 = derivative(v1, s)   # v''(s),  shape (60, 1)
v3 = derivative(v2, s)   # v'''(s), shape (60, 1)
v4 = derivative(v3, s)   # v''''(s), shape (60, 1)

```

Each call differentiates the previous result with respect to the input coordinates s. The computational graph grows deeper with each call, which is why `create_graph=True` is mandatory in the derivative helper.

**Step 4: PDE loss**

```python
loss_pde = torch.mean((v4 - 1)**2)

```

The nondimensional beam equation is v''''(s) = 1. The loss measures the mean squared deviation of the computed 4th derivative from unity across all 60 collocation points.

**Step 5: Free-end boundary conditions**

```python
vL = model(sL)              # v(1)
v1L = derivative(vL, sL)    # v'(1)
v2L = derivative(v1L, sL)   # v''(1)
v3L = derivative(v2L, sL)   # v'''(1)

```

We evaluate the network at s=1 and differentiate up to 3rd order. The conditions v''(1)=0 and v'''(1)=0 correspond to zero bending moment and zero shear at the free end.

```python
loss_free = torch.mean(v2L**2 + v3L**2)

```

Both conditions are penalized in a single scalar loss term.

**Step 6: Total loss with weighting**

```python
loss = loss_pde + 10 * loss_free

```

The BC loss is weighted by 10. This reflects the importance of satisfying boundary conditions -- the solution quality depends critically on correct BC enforcement. Without this weighting, the optimizer might reduce PDE residual while leaving BCs poorly satisfied, especially since v''(1) and v'''(1) are higher derivatives that the network finds harder to control precisely.

**Why weight=10 specifically:**

The PDE loss is averaged over 60 points while the BC loss covers just one point (with two terms). The factor of 10 ensures the boundary conditions compete effectively against the bulk PDE residual. Too large a weight would over-prioritize BCs at the expense of interior accuracy; 10 provides a good balance for this problem.

**Step 7: Backpropagation and update**

`loss.backward()` computes gradients of the total loss with respect to all network parameters through the entire 4-derivative chain. `optimizer.step()` applies the Adam update. `scheduler.step()` decrements the LR counter.

**Step 8: Best-state tracking**

The model state dict is saved whenever a new minimum loss is achieved. This guards against late-training oscillations that can arise when the scheduler halves the learning rate, causing temporary instability.

---

## Section 10: Post-Training

```python
model.load_state_dict(best_state)

```

After 8000 epochs, the best state (lowest total loss observed during training) is loaded back into the model. This ensures evaluation uses the optimal parameters rather than whatever state the model happened to end in.

**Why this matters for beam problems:**

The 4th derivative is highly sensitive to small parameter perturbations. A model state with slightly lower PDE loss but worse BC loss (or vice versa) might produce visibly worse deflection predictions. Best-state selection provides the most balanced result.

---

## Section 11: Evaluation and Physical Conversion

```python
s_test = torch.linspace(0, 1, 200, device=device).reshape(-1, 1)
with torch.no_grad():
    v_pred = model(s_test).cpu().numpy()

x_test = s_test.cpu().numpy() * L
w_pred = w_scale * v_pred
w_exact = exact_deflection(x_test)

```

**Step-by-step:**

1. **Test grid:** 200 evenly spaced points in [0,1], finer than training (60). No `requires_grad` needed since we only evaluate, not differentiate.
2. **Inference:** `torch.no_grad()` disables gradient tracking for speed. The model outputs nondimensional v(s) values.
3. **Physical conversion:**- `x_test = s_test * L` converts nondimensional coordinate to meters.
- `w_pred = w_scale * v_pred` converts nondimensional deflection to meters.
- `w_scale = q*L^4/EI = 6.0e-4 m` is the scaling factor from Section 3.
4. **Exact solution:** `exact_deflection(x_test)` computes the analytical result at the same physical coordinates.

**Metrics computed:**

```python
tip_deflection = w_pred[-1]          # Deflection at x=L
max_error = np.max(np.abs(w_pred - w_exact))
relative_error = max_error / np.max(np.abs(w_exact))

```

- **Tip deflection:** The most important engineering quantity. Exact value is qL^4/(8EI) = 7.5e-5 m.
- **Max absolute error:** Worst-case pointwise deviation in meters.
- **Relative error:** Normalized by the maximum exact deflection (at the tip). Typical values after 8000 epochs: < 0.1%.

---

## Section 12: Visualization

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: Deflection profile
ax1.plot(x_test, w_exact, 'b-', label='Exact')
ax1.plot(x_test, w_pred, 'r--', label='PINN')
ax1.set_xlabel('x [m]')
ax1.set_ylabel('w [m]')
ax1.set_title('Cantilever Beam Deflection')
ax1.legend()

# Panel 2: Training convergence
ax2.semilogy(losses)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Training Convergence')

plt.tight_layout()
plt.savefig('beam_pinn_results.png', dpi=150)
plt.show()

```

**Panel 1 -- Deflection profile:**

Shows the beam deflection w(x) from fixed end (x=0, w=0) to free end (x=L). The exact solution is a quartic polynomial with maximum deflection at the tip. The PINN prediction (dashed red) should overlay the exact solution (solid blue) with negligible visible deviation.

**Panel 2 -- Training convergence:**

Logarithmic y-axis shows loss decrease over 8000 epochs. Expected features:

- Rapid initial decrease (epochs 0-500) as the network finds the solution basin.
- Steady improvement with occasional plateaus (epochs 500-4000).
- Visible step at epoch 4000 when LR halves (loss may spike briefly then resume descent).
- Final convergence to O(10^-6) or lower.

---

## Comparison Across PINN Projects

| Aspect | Thermoelastic Bar | Channel Flow | Cantilever Beam |
| --- | --- | --- | --- |
| PDE order | 2nd order | 2nd order | 4th order |
| Governing equation | u'' = -alpha*T(x) | u'' = (1/mu)*dp/dx | EI*w'''' = q |
| Domain dimension | 1D | 1D | 1D |
| BC type | Dirichlet (both ends) | Dirichlet (both ends) | Mixed (fixed + free) |
| Hard constraint | (1-x)*x*N(x) | (1-y^2)*N(y) | s^2 * N(s) |
| BCs enforced hard | u(0)=0, u(1)=0 | u(-1)=0, u(1)=0 | w(0)=0, w'(0)=0 |
| BCs enforced soft | None | None | w''(L)=0, w'''(L)=0 |
| Soft BC weight | N/A | N/A | 10 |
| Derivatives needed | 2 | 2 | 4 |
| Activation | tanh | tanh | tanh |
| Hidden layers | 3 x 40 | 3 x 32 | 3 x 30 |
| Collocation points | 50 | 50 | 60 |
| Training epochs | 10000 | 10000 | 8000 |
| Optimizer | Adam | Adam | Adam |
| Scheduler | StepLR | StepLR | StepLR |
| Nondim scale | alpha*T0*L^2 | (dp/dx)*H^2/mu | q*L^4/EI |
| Physical output | Displacement [m] | Velocity [m/s] | Deflection [m] |

**Key differences explained:**

1. **4th order vs 2nd order:** The beam problem requires four derivative calls instead of two. This creates a deeper computational graph and makes training more sensitive to learning rate and network architecture choices.
2. **Mixed boundary conditions:** Unlike the bar and channel problems where all boundaries have the same type (Dirichlet), the beam has two Dirichlet-like conditions at x=0 and two Neumann-like conditions at x=L. This asymmetry demands the split strategy: hard constraints at one end, soft penalties at the other.
3. **Hard constraint design:** The s^2 prefactor enforces a double zero (both value and slope) at s=0. This is more sophisticated than (1-x)*x which enforces simple zeros at both endpoints, or (1-y^2) which enforces zeros at y= +/-1.
4. **Soft BC necessity:** The free-end conditions involve v''(L) and v'''(L). There is no simple algebraic prefactor that can guarantee these higher-order derivative conditions while maintaining network expressiveness. Hence they must be enforced through the loss function with appropriate weighting.
5. **Fewer epochs:** Despite higher PDE order, 8000 epochs suffice because the target solution is a smooth low-degree polynomial (quartic). The network architecture with tanh activations naturally represents such smooth functions efficiently.

---

## Summary of Design Decisions

| Decision | Rationale |
| --- | --- |
| s^2 prefactor (not s or s^3) | Exactly matches the two fixed-end conditions |
| Soft BC with weight 10 | Free-end conditions involve high derivatives |
| 30 neurons per layer | Balances expressiveness vs 4th-derivative cost |
| tanh (not ReLU/GELU) | C-infinity smoothness required for 4th derivative |
| 60 collocation points | Adequate density for quartic polynomial target |
| StepLR at epoch 4000 | Two-phase: explore then refine high-order features |
| Separate sL tensor | Clean separation of PDE and BC loss computation |
| w_scale nondimensionalization | Keeps all quantities O(1), essential for 4th order |
| Best-state tracking | Guards against 4th-derivative sensitivity |
| 200 test points | Finer than training grid for honest evaluation |

---

## Computational Graph Depth

For a single forward-backward pass, the graph depth scales with derivative order:

```
Network forward:  depth ~ 4 (layers)
1st derivative:   depth ~ 4 + 4 = 8
2nd derivative:   depth ~ 8 + 4 = 12
3rd derivative:   depth ~ 12 + 4 = 16
4th derivative:   depth ~ 16 + 4 = 20

Backprop through all:  depth ~ 40 (reverse pass)

```

This deep graph explains why:

- Training is slower per epoch than 2nd-order PINNs.
- Smooth activations (tanh) are essential to avoid dead paths.
- Moderate network width (30) controls memory and computation.
- The network still trains successfully because the target is smooth.

---

## End-to-End Data Flow

```
Physical problem:  EI*w''''(x) = q,  w ~ O(10^-4) m
                          |
                   Nondimensionalize
                          |
                          v
Nondim problem:    v''''(s) = 1,  v ~ O(1)
                          |
                   Neural network
                          |
                          v
Network output:    v(s) = s^2 * N(s)  [enforces v(0)=v'(0)=0]
                          |
                   4x autograd.grad
                          |
                          v
PDE residual:      (v'''' - 1)^2  at 60 points
BC residual:       v''(1)^2 + v'''(1)^2
                          |
                   Adam optimization (8000 epochs)
                          |
                          v
Trained model:     v_pred(s) at 200 test points
                          |
                   Rescale: w = w_scale * v
                          |
                          v
Physical result:   w(x) in meters, compared to exact solution

```

