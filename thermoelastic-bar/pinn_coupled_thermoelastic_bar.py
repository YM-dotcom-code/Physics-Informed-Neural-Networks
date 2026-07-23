import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
 
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
# === COUPLED THERMOELASTIC PROBLEM ===
# Heat equation (steady): k * d²T/dx² = 0, T(0)=100, T(L)=500
# Elasticity with thermal strain: E*d²u/dx² + E*alpha*dT/dx = 0
#   BC: u(0)=0 (fixed end), sigma(L)=0 → du/dx(L) = alpha*(T(L)-T_ref)
# Parameters:
k = 50.0       # Thermal conductivity [W/mK]
E_mod = 200e9  # Young's modulus [Pa]
alpha = 12e-6  # Thermal expansion coefficient [1/K]
L = 1.0        # Bar length [m]
T0, TL = 100.0, 500.0  # Temperature BCs
T_ref = 100.0  # Reference temperature (stress-free)
 
# Analytical solutions:
# T(x) = T0 + (TL-T0)*x/L  (linear, since d²T/dx²=0)
# u(x) = alpha * (T(x) - T_ref) * x - alpha*(TL-T0)*x²/(2L)
#       = alpha*(TL-T0)/(2L) * (2Lx - x²) ... simplified
# Actually: from E*u'' = -E*alpha*T' and T'=(TL-T0)/L:
#   u'' = -alpha*(TL-T0)/L
#   u(x) = -alpha*(TL-T0)/(2L) * x² + C1*x + C2
#   u(0)=0 → C2=0
#   u'(L) = alpha*(T(L)-T_ref) = alpha*(TL-T_ref)
#   u'(L) = -alpha*(TL-T0)/L * L + C1 = -alpha*(TL-T0) + C1
#   C1 = alpha*(TL-T_ref) + alpha*(TL-T0) = alpha*(2*TL - T_ref - T0)
#   Since T_ref=T0: C1 = alpha*(2*TL - 2*T0) = 2*alpha*(TL-T0)
def exact_T(x):
    return T0 + (TL - T0) * x / L
 
def exact_u(x):
    dT_dx = (TL - T0) / L
    C1 = alpha * (2*(TL - T0))
    return -alpha * dT_dx / 2 * x**2 + C1 * x
 
# === NETWORK: 1 input → 2 outputs [T, u] ===
class ThermoElasticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(1, 40), nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
        )
        self.head_T = nn.Linear(40, 1)  # Temperature output
        self.head_u = nn.Linear(40, 1)  # Displacement output
    def forward(self, x):
        h = self.shared(x)
        return self.head_T(h), self.head_u(h)
 
model = ThermoElasticNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
 
# === COLLOCATION ===
n_int = 100
x_int = torch.linspace(0, L, n_int, device=device).reshape(-1, 1)
x_int.requires_grad_(True)
x0 = torch.tensor([[0.0]], device=device, requires_grad=True)
xL_t = torch.tensor([[L]], device=device, requires_grad=True)
 
# === TRAINING ===
print('Training coupled thermoelastic PINN...\n')
losses = []
for epoch in range(15000):
    optimizer.zero_grad()
    # Forward pass
    T, u = model(x_int)
    # --- Heat equation: k*T'' = 0 ---
    T_x = torch.autograd.grad(T, x_int, torch.ones_like(T), create_graph=True)[0]
    T_xx = torch.autograd.grad(T_x, x_int, torch.ones_like(T_x), create_graph=True)[0]
    res_heat = k * T_xx
    loss_heat = torch.mean(res_heat**2)
    # --- Elasticity: E*u'' + E*alpha*T' = 0 ---
    u_x = torch.autograd.grad(u, x_int, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_int, torch.ones_like(u_x), create_graph=True)[0]
    res_elast = E_mod * u_xx + E_mod * alpha * T_x
    loss_elast = torch.mean(res_elast**2) / E_mod**2  # Normalize
    # --- Boundary conditions ---
    T_0, u_0 = model(x0)
    T_L, u_L = model(xL_t)
    # T BCs
    loss_T_bc = (T_0 - T0)**2 + (T_L - TL)**2
    # u BC: u(0)=0
    loss_u_bc = u_0**2
    # Stress-free at x=L: du/dx(L) = alpha*(T(L) - T_ref)
    u_L_val = model(xL_t)[1]
    u_x_L = torch.autograd.grad(u_L_val, xL_t, torch.ones_like(u_L_val), create_graph=True)[0]
    target_strain = alpha * (TL - T_ref)
    loss_stress_bc = (u_x_L - target_strain)**2
    # Combined loss
    loss_bc = loss_T_bc.squeeze() + 1e12*loss_u_bc.squeeze() + 1e12*loss_stress_bc.squeeze()
    loss = loss_heat + loss_elast + 100*loss_bc
    loss.backward()
    optimizer.step()
    scheduler.step()
    losses.append(loss.item())
    if (epoch + 1) % 5000 == 0:
        print(f'  Epoch {epoch+1}, Loss: {loss.item():.2e}')
 
# === EVALUATION ===
x_test = torch.linspace(0, L, 200, device=device).reshape(-1, 1)
with torch.no_grad():
    T_pred, u_pred = model(x_test)
    T_pred = T_pred.cpu().numpy().flatten()
    u_pred = u_pred.cpu().numpy().flatten()
x_np = x_test.cpu().numpy().flatten()
T_ex = exact_T(x_np)
u_ex = exact_u(x_np)
 
print(f'\n=== Results ===')
print(f'Temperature: max error = {np.max(np.abs(T_pred-T_ex)):.3f}°C')
print(f'Displacement: max error = {np.max(np.abs(u_pred-u_ex))*1e6:.3f} μm')
print(f'Max displacement (exact): {np.max(u_ex)*1e6:.1f} μm')
print(f'Max displacement (PINN):  {np.max(u_pred)*1e6:.1f} μm')
 
# === PLOT ===
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# Temperature
axes[0].plot(x_np, T_ex, 'b-', linewidth=2, label='Exact')
axes[0].plot(x_np, T_pred, 'r--', linewidth=2, label='PINN')
axes[0].set_xlabel('x [m]'); axes[0].set_ylabel('T [°C]')
axes[0].set_title('Temperature Field'); axes[0].legend()
# Displacement
axes[1].plot(x_np, u_ex*1e6, 'b-', linewidth=2, label='Exact')
axes[1].plot(x_np, u_pred*1e6, 'r--', linewidth=2, label='PINN')
axes[1].set_xlabel('x [m]'); axes[1].set_ylabel('u [μm]')
axes[1].set_title('Displacement (Thermal Expansion)'); axes[1].legend()
# Loss
axes[2].semilogy(losses)
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Loss')
axes[2].set_title('Training Convergence')
plt.tight_layout()
plt.savefig('thermoelastic.png', dpi=150)
plt.show()
print('\nKey insight: One network, two outputs, coupled physics.')
print('The thermal field drives the mechanical response automatically.')
