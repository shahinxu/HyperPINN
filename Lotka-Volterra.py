from HyperPINNWeight import HyperPINNWeight
import torch
from torch import nn
from torch import optim
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import lstsq
from sympy import symbols, Matrix, lambdify
    
def LVeqComplete(t, Npop, A, R, K, TriangleList):
    coup_rete = A @ Npop
    term1 = Npop * R * (1 - Npop / K)
    term2 = Npop * coup_rete
    coup_simplicial = np.zeros_like(Npop)
    for row in TriangleList:
        i1, i2, i3, b = row
        i1, i2, i3 = int(i1), int(i2), int(i3)  
        coup_simplicial[i1] += b * Npop[i2] * Npop[i3]
    term3 = Npop * coup_simplicial
    return term1+term2+term3

R = np.array([0.6099, 0.6177, 0.8594, 0.8055, 0.5767, 0.1829, 0.2399])
K = np.array([88.7647, 3.8387, 49.5002, 17.6248, 97.8894, 71.5568, 50.5467])
N = 7
A = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.8, 0, 0],
    [0, 0.1, 0, 0, 0, 0, 0],
    [-0.4, 0, 0, 0, 0, -0.3, 0],
    [0, 0, 0, 0.7, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -0.7],
    [0, -0.8, 0, 0.2, 0, 0, 0]], dtype=float)
for i in range(N):
    A[i, :] *= R[i] / K[i]
TriangleList = np.array([[1, 2, 6, 0.0062],[3, 0, 5, 0.0016]])
Tri_values = TriangleList[:, -1]
x0 = np.random.uniform(30, 50, size=(N,))
M = 150
tmax = 15
dt = tmax / M
t_eval = np.linspace(0, tmax, M+1)
t_data = torch.linspace(0, tmax, M+1, requires_grad=True).unsqueeze(1) 
sol = solve_ivp(LVeqComplete, (0, tmax), x0, method='RK45',t_eval=t_eval, args=(A, R, K, TriangleList),rtol=1e-9, atol=1e-9)
X = sol.y.T 

## Add noise
noise_level = 0.05 * np.std(sol.y) 
X += noise_level * np.random.randn(*X.shape)
x_data = torch.tensor(X, dtype=torch.float64)
A_mask = torch.tensor(A != 0, dtype=torch.float64)

model = HyperPINNWeight(input_dim=1, hidden_dim=128, output_dim=N, A_mask=A_mask)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)
epochs = 40000
warmup_epochs = 7000
transition_epochs = 7000

for epoch in range(epochs):
    optimizer.zero_grad()
    x_pred, A_pred = model(t_data)
    loss_data = torch.mean((x_pred - x_data)**2)
    loss_physics = model.physics_loss(t_data)
    x_init, _ = model(torch.tensor([[0.0]]))
    loss_init = torch.mean((x_init - x_data[0:1]) ** 2)
    if epoch < warmup_epochs:
        loss = loss_data + loss_init
    elif epoch < warmup_epochs + transition_epochs:
        progress = (epoch - warmup_epochs) / transition_epochs
        physics_weight = progress
        loss = loss_data + loss_init + physics_weight * loss_physics
    else:
        loss = loss_data + loss_init + 1.0 * loss_physics
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()    
    if epoch > 70000:
        if epoch % 2000 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            new_lr = current_lr * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(new_lr, 1e-6)
            print(f"Learning rate adjusted to: {new_lr}")
    A_true = A
    A_est = model.get_structured_A().detach().numpy()
    Tri1_est = model.Tri1.item()
    Tri2_est = model.Tri2.item()
    error = np.sqrt((np.linalg.norm(A_true - A_est) ** 2 + (Tri_values[0] - Tri1_est) ** 2
         + (Tri_values[1] - Tri2_est) ** 2) / (np.linalg.norm(A_true) ** 2 + Tri_values[0] ** 2 + Tri_values[1] ** 2))
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}")
        print(f"Total Loss: {loss.item():.6f}")
        print(f"Data Loss: {loss_data.item():.6f}")
        print(f"Physics Loss: {loss_physics.item():.6f}")
        print(f"Error: {error:.6f}")
        print("---")

target_loss = 1e-6
epoch_1, epoch_2, epoch_3 = 2000, 5000, 5000
A_true = A
best_loss = float('inf')
loss_history = []
def run_stage(*, stage_name, epochs, build_loss, make_opt, make_sched=None,
              log_every=500, early_stop=None):
    global best_loss
    optimizer = make_opt(model)
    scheduler = make_sched(optimizer) if make_sched else None
    consecutive_small_improvements = 0  

    for epoch in range(epochs):
        optimizer.zero_grad()
        x_pred, A_pred = model(t_data)
        loss_data = torch.mean((x_pred - x_data) ** 2)
        loss_physics = model.physics_loss(t_data)
        t_init = torch.tensor([[0.0]], dtype=t_data.dtype, device=t_data.device)
        x_init, _ = model(t_init)
        loss_init = torch.mean((x_init - x_data[0:1]) ** 2)
        total_loss, clip_mode = build_loss(epoch, loss_data, loss_init, loss_physics)
        total_loss.backward()
        if clip_mode == "smart":  # Stage 1
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            if grad_norm > 5.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            elif grad_norm > 0.5:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        elif clip_mode == "mild":  # Stage 2
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            if total_grad_norm > 0.01:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
        else:
            pass  
        optimizer.step()
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(total_loss.item())
            else:
                scheduler.step()

        current_loss = total_loss.item()
        loss_history.append(current_loss)
        if current_loss < best_loss:
            best_loss = current_loss
        if stage_name == "Stage 2" and len(loss_history) >= 2:
            improvement = loss_history[-2] - current_loss
            consecutive_small_improvements = (consecutive_small_improvements + 1) if (improvement < 1e-12) else 0
            if epoch % 1000 == 0:
                lr = optimizer.param_groups[0]['lr']
                with torch.no_grad():
                    Tri1_est = model.Tri1.item()
                    Tri2_est = model.Tri2.item()
                    A_est = model.get_structured_A().detach().numpy()
                error = np.sqrt((np.linalg.norm(A_true - A_est) ** 2 + (Tri_values[0] - Tri1_est) ** 2
                     + (Tri_values[1] - Tri2_est) ** 2) / (np.linalg.norm(A_true) ** 2 + Tri_values[0] ** 2 + Tri_values[1] ** 2))
                print(f"{stage_name} Epoch {epoch}: Loss={current_loss:.10f}, LR={lr:.2e}")
                print(f"  Error={error:.8f}")
                print(f"  Grad Norm={total_grad_norm:.2e}")

            if consecutive_small_improvements > 1000:
                print(f"Adding perturbation at epoch {epoch}")
                with torch.no_grad():
                    for p in model.parameters():
                        p.add_(torch.randn_like(p) * 1e-6)
                consecutive_small_improvements = 0

        if stage_name == "Stage 1" and epoch % 500 == 0:
            lr = optimizer.param_groups[0]['lr']
            with torch.no_grad():
                tri1_val = model.Tri1.item()
                tri2_val = model.Tri2.item()
            print(f"{stage_name} Epoch {epoch}: Loss={current_loss:.8f}, LR={lr:.2e}")
            print(f"  Tri1={tri1_val:.6f}, Tri2={tri2_val:.6f}")
        if stage_name == "Stage 3" and epoch % 500 == 0:
            with torch.no_grad():
                Tri1_est = model.Tri1.item()
                Tri2_est = model.Tri2.item()
                A_est = model.get_structured_A().detach().numpy()
                error = np.sqrt((np.linalg.norm(A_true - A_est) ** 2 + (Tri_values[0] - Tri1_est) ** 2
                     + (Tri_values[1] - Tri2_est) ** 2) / (np.linalg.norm(A_true) ** 2 + Tri_values[0] ** 2 + Tri_values[1] ** 2))
            print(f"{stage_name} Epoch {epoch}: Loss={current_loss:.12f}")
            print(f"  Error={error:.8f}")
        if (early_stop is not None) and (current_loss < early_stop):
            print(f"{stage_name} completed at epoch {epoch}, loss: {current_loss:.8f}")
            break
            
print("=== Stage 1: Medium Precision Optimization ===")
run_stage(stage_name="Stage 1",epochs=epoch_1,
    build_loss=lambda ep, ld, li, lp: (ld + 2.0 * li + 0.5 * lp, "smart"),
    make_opt=lambda m: optim.Adam(m.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-12),
    make_sched=lambda opt: optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.3, patience=200, threshold=1e-10, min_lr=1e-12),
    log_every=500,early_stop=target_loss * 100)
print("\n=== Stage 2: Ultra High Precision Optimization ===")
def stage2_build_loss(epoch, ld, li, lp):
    return ((ld + 3.0 * li + 0.3 * lp) if epoch < 2000 else (ld + 1.0 * li + 0.1 * lp), "mild")

run_stage(
    stage_name="Stage 2",epochs=epoch_2,build_loss=stage2_build_loss,
    make_opt=lambda m: optim.Adam(m.parameters(), lr=1e-5, betas=(0.99, 0.9999), eps=1e-16),
    make_sched=lambda opt: optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1000, T_mult=2, eta_min=1e-10),
    log_every=1000,early_stop=target_loss)
if best_loss > target_loss:
    print("\n=== Stage 3: Final Fine-tuning ===")
    run_stage(stage_name="Stage 3",epochs=epoch_3,build_loss=lambda ep, ld, li, lp: (ld + li + 0.01 * lp, None),
        make_opt=lambda m: optim.Adam(m.parameters(), lr=1e-8, betas=(0.999, 0.99999), eps=1e-20),make_sched=None,
        log_every=500,early_stop=target_loss)
print(f"\nFinal Results:")
print(f"Best Loss: {best_loss:.12f}")
print(f"Target Loss: {target_loss}")

with torch.no_grad():
    A_est = model.get_structured_A().detach().cpu().numpy()
    Tri1_est = model.Tri1.item()
    Tri2_est = model.Tri2.item()

final_error = np.sqrt((np.linalg.norm(A_true - A_est) ** 2 + (Tri_values[0] - Tri1_est) ** 2 + (Tri_values[1] - Tri2_est) ** 2) /
    (np.linalg.norm(A_true) ** 2 + Tri_values[0] ** 2 + Tri_values[1] ** 2))
print("Final error:", final_error)
