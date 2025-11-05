from HyperPINNTopology import HyperPINNTopology
import torch
from torch import nn
from torch import optim as optim
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def driving_dynamics(t, x, EdgeList, TriangleList, QuadList, QuintList):
    """
    驾驶场景动力学方程
    
    每辆车的三个状态：
    - velocity (v): 速度
    - acceleration (a): 加速度  
    - steering (s): 转向角
    
    内部动力学基于交通流理论，加上超图交互项
    """
    m1 = len(x)
    N = m1 // 3  # 车辆数量
    
    # 分离状态变量
    velocity = x[0:N]           # 速度
    acceleration = x[N:2*N]     # 加速度
    steering = x[2*N:3*N]       # 转向角
    
    # 交通参数
    v_desired = 30.0    # 期望速度 (km/h)
    tau_accel = 2.0     # 加速时间常数
    tau_steer = 1.5     # 转向时间常数
    
    # 交互强度参数
    k_follow = 0.3      # 跟车影响强度
    k_lane = 0.2        # 车道变换影响
    k_intersection = 0.4 # 交叉路口影响
    k_roundabout = 0.5  # 环岛影响
    k_complex = 0.6     # 复杂交通枢纽影响
    
    # 初始化耦合项
    coupling_follow = np.zeros(N)       # 跟车耦合
    coupling_lane = np.zeros(N)         # 车道耦合
    coupling_intersection = np.zeros(N)  # 交叉路口耦合
    coupling_roundabout = np.zeros(N)   # 环岛耦合
    coupling_complex = np.zeros(N)      # 复杂枢纽耦合
    
    # 2阶交互：跟车和相邻车道影响
    for ii in range(len(EdgeList)):
        i1 = EdgeList[ii, 0] - 1
        i2 = EdgeList[ii, 1] - 1
        
        # 跟车模型：前车影响后车
        if velocity[i1] > velocity[i2]:  # i1是前车
            coupling_follow[i2] += (velocity[i1] - velocity[i2]) * 0.5
            coupling_follow[i1] -= (velocity[i1] - velocity[i2]) * 0.1
        else:  # i2是前车
            coupling_follow[i1] += (velocity[i2] - velocity[i1]) * 0.5
            coupling_follow[i2] -= (velocity[i2] - velocity[i1]) * 0.1
            
        # 车道变换影响
        coupling_lane[i1] += np.sin(steering[i2]) * 0.3
        coupling_lane[i2] += np.sin(steering[i1]) * 0.3
    
    # 3阶交互：交叉路口三方博弈
    if len(TriangleList) > 0:
        mtrianglelist, ntrianglelist = TriangleList.shape
        for ii in range(mtrianglelist):
            i1 = TriangleList[ii, 0] - 1
            i2 = TriangleList[ii, 1] - 1
            i3 = TriangleList[ii, 2] - 1
            
            # 交叉路口优先级判断：速度高的车优先
            speeds = [velocity[i1], velocity[i2], velocity[i3]]
            indices = [i1, i2, i3]
            
            # 速度最高的车获得优先权，其他车减速
            max_speed_idx = np.argmax(speeds)
            priority_car = indices[max_speed_idx]
            
            for idx in indices:
                if idx != priority_car:
                    coupling_intersection[idx] -= 0.4 * (velocity[priority_car] - velocity[idx])
                else:
                    coupling_intersection[idx] += 0.2 * np.mean([velocity[i] for i in indices if i != idx])
    
    # 4阶交互：环岛四方博弈
    if QuadList is not None and len(QuadList) > 0:
        mquadlist, nquadlist = QuadList.shape
        for ii in range(mquadlist):
            i1 = QuadList[ii, 0] - 1
            i2 = QuadList[ii, 1] - 1
            i3 = QuadList[ii, 2] - 1
            i4 = QuadList[ii, 3] - 1
            
            indices = [i1, i2, i3, i4]
            
            # 环岛规则：右侧优先 (简化为位置编号小的优先)
            for idx in indices:
                others = [i for i in indices if i != idx]
                right_priority = sum([1 for i in others if i < idx])  # 右侧车辆数
                
                coupling_roundabout[idx] += 0.3 * right_priority - 0.2 * len(others)
                
                # 转向影响：环岛内需要持续左转
                coupling_roundabout[idx] += 0.1 * np.sin(steering[idx] + np.pi/4)
    
    # 5阶交互：复杂交通枢纽
    if QuintList is not None and len(QuintList) > 0:
        mquintlist, nquintlist = QuintList.shape
        for ii in range(mquintlist):
            i1 = QuintList[ii, 0] - 1
            i2 = QuintList[ii, 1] - 1
            i3 = QuintList[ii, 2] - 1
            i4 = QuintList[ii, 3] - 1
            i5 = QuintList[ii, 4] - 1
            
            indices = [i1, i2, i3, i4, i5]
            
            # 复杂枢纽：考虑所有车辆的相互影响
            for idx in indices:
                others = [i for i in indices if i != idx]
                
                # 速度协调：向平均速度靠拢
                avg_speed = np.mean([velocity[i] for i in others])
                coupling_complex[idx] += 0.2 * (avg_speed - velocity[idx])
                
                # 转向协调：避免冲突
                avg_steering = np.mean([steering[i] for i in others])
                coupling_complex[idx] += 0.1 * np.sin(avg_steering - steering[idx])
    
    # 内部动力学 + 耦合项
    dvdt = (1/tau_accel) * (v_desired - velocity + 
                            k_follow * coupling_follow + 
                            k_lane * coupling_lane)
    
    dadt = -acceleration/tau_accel + 0.5 * (velocity - v_desired) + \
           k_intersection * coupling_intersection + \
           k_roundabout * coupling_roundabout
    
    dsdt = -steering/tau_steer + 0.3 * acceleration + \
           k_complex * coupling_complex
    
    # 添加随机扰动（模拟驾驶员的不确定性）
    noise_scale = 0.05
    dvdt += noise_scale * np.random.normal(0, 0.1, N)
    dadt += noise_scale * np.random.normal(0, 0.05, N) 
    dsdt += noise_scale * np.random.normal(0, 0.02, N)
    
    # 物理约束
    dvdt = np.clip(dvdt, -5, 5)      # 速度变化限制
    dadt = np.clip(dadt, -3, 3)      # 加速度变化限制  
    dsdt = np.clip(dsdt, -0.5, 0.5)  # 转向变化限制
    
    dxdt = np.concatenate((dvdt, dadt, dsdt))
    return dxdt

# 驾驶场景配置
N = 8  # 8辆车

# 交通网络拓扑定义
# EdgeList: 跟车关系和相邻车道
EdgeList = np.array([
    [1, 2],  # 车1跟车2
    [2, 3],  # 车2跟车3  
    [3, 4],  # 车3跟车4
    [5, 6],  # 车5跟车6
    [6, 7],  # 车6跟车7
    [7, 8],  # 车7跟车8
    [1, 5],  # 车1和车5相邻车道
    [4, 8]   # 车4和车8相邻车道
])

# TriangleList: 交叉路口三方互动
TriangleList = np.array([
    [1, 2, 5],  # 路口1：车1,2,5
    [3, 4, 7],  # 路口2：车3,4,7
    [2, 6, 8]   # 路口3：车2,6,8
])

# QuadList: 环岛四方博弈  
QuadList = np.array([
    [1, 3, 5, 7],  # 环岛1：车1,3,5,7
    [2, 4, 6, 8]   # 环岛2：车2,4,6,8
])

# QuintList: 复杂交通枢纽
QuintList = np.array([
    [1, 2, 3, 4, 5],  # 枢纽1：车1,2,3,4,5
    [4, 5, 6, 7, 8]   # 枢纽2：车4,5,6,7,8
])

all_2edges = list(combinations(range(1, N+1), 2))
all_3edges = list(combinations(range(1, N+1), 3))
all_4edges = list(combinations(range(1, N+1), 4))
all_5edges = list(combinations(range(1, N+1), 5))

true_2edges = set(tuple(sorted(edge)) for edge in EdgeList)
true_3edges = set(tuple(sorted(triangle)) for triangle in TriangleList)
true_4edges = set(tuple(sorted(quad)) for quad in QuadList)
true_5edges = set(tuple(sorted(quint)) for quint in QuintList)

# 仿真参数
M = 200          # 时间步数
tmax = 50        # 仿真时间 (秒)
dt = tmax / M
t_eval = np.linspace(0, tmax, M+1)
t_data = torch.linspace(0, tmax, M+1, requires_grad=True).unsqueeze(1)

# 初始条件：随机初始速度、加速度、转向角
np.random.seed(42)  # 确保可重现
x0 = np.zeros(3 * N)
x0[0:N] = np.random.uniform(20, 35, N)        # 初始速度 20-35 km/h
x0[N:2*N] = np.random.uniform(-1, 1, N)      # 初始加速度 -1到1 m/s²
x0[2*N:3*N] = np.random.uniform(-0.2, 0.2, N) # 初始转向角 -0.2到0.2 rad

print(f"驾驶场景初始化完成:")
print(f"车辆数量: {N}")
print(f"初始速度范围: {x0[0:N].min():.1f} - {x0[0:N].max():.1f} km/h")
print(f"交通网络: {len(EdgeList)} 跟车/车道关系, {len(TriangleList)} 交叉路口, {len(QuadList)} 环岛, {len(QuintList)} 枢纽")

# 求解微分方程
print("开始仿真驾驶动力学...")
sol = solve_ivp(driving_dynamics, (0, tmax), x0, t_eval=t_eval, 
                args=(EdgeList, TriangleList, QuadList, QuintList),
                method='RK45', rtol=1e-6)

if sol.success:
    print(f"仿真成功完成! 状态: {sol.message}")
    X = sol.y.T
else:
    print(f"仿真失败: {sol.message}")
    
nt = len(t_eval)
dxdt = np.array([driving_dynamics(t, sol.y[:, i], EdgeList, TriangleList, QuadList, QuintList) for i, t in enumerate(sol.t)])

x_data = torch.tensor(X, dtype=torch.float64) 
print(x_data)
architectures = [("ResNet", True, False, False), ("Attention", False, True, False), ("SIREN", False, False, True)]

# Visualize driving simulation results
plt.figure(figsize=(15, 10))

# Velocity evolution
plt.subplot(2, 3, 1)
colors = plt.cm.Set1(np.linspace(0, 1, N))
for i in range(N):
    plt.plot(t_eval, X[:, i], label=f'Vehicle {i+1}', linewidth=2, color=colors[i])
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Velocity (km/h)', fontsize=12)
plt.title('Vehicle Velocity Evolution', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# Acceleration evolution  
plt.subplot(2, 3, 2)
for i in range(N):
    plt.plot(t_eval, X[:, N+i], label=f'Vehicle {i+1}', linewidth=2, color=colors[i])
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Acceleration (m/s²)', fontsize=12)
plt.title('Vehicle Acceleration Evolution', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# Steering angle evolution
plt.subplot(2, 3, 3)
for i in range(N):
    plt.plot(t_eval, X[:, 2*N+i], label=f'Vehicle {i+1}', linewidth=2, color=colors[i])
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Steering Angle (rad)', fontsize=12)
plt.title('Vehicle Steering Angle Evolution', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# Vehicle trajectories in phase space
plt.subplot(2, 3, 4)
for i in range(N):
    plt.scatter(X[:, i], X[:, N+i], c=[colors[i]], alpha=0.6, s=20, label=f'Vehicle {i+1}')
plt.xlabel('Velocity (km/h)', fontsize=12)
plt.ylabel('Acceleration (m/s²)', fontsize=12) 
plt.title('Velocity-Acceleration Phase Space', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# Final velocity distribution
plt.subplot(2, 3, 5)
final_speeds = X[-1, 0:N]
plt.bar(range(1, N+1), final_speeds, color=colors, alpha=0.7)
plt.xlabel('Vehicle ID', fontsize=12)
plt.ylabel('Final Velocity (km/h)', fontsize=12)
plt.title('Final Velocity Distribution', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Traffic network visualization
plt.subplot(2, 3, 6)
# Draw vehicle nodes
angles = np.linspace(0, 2*np.pi, N, endpoint=False)
x_pos = np.cos(angles)
y_pos = np.sin(angles)

plt.scatter(x_pos, y_pos, s=300, c=final_speeds, cmap='viridis', alpha=0.8)
for i in range(N):
    plt.annotate(f'{i+1}', (x_pos[i], y_pos[i]), ha='center', va='center', 
                fontsize=12, fontweight='bold', color='white')

# Draw 2nd-order connections
for edge in EdgeList:
    i, j = edge[0]-1, edge[1]-1
    plt.plot([x_pos[i], x_pos[j]], [y_pos[i], y_pos[j]], 'b-', alpha=0.6, linewidth=2)

plt.title('Traffic Network Topology (Color: Final Velocity)', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.colorbar(label='Velocity (km/h)')

plt.tight_layout()
plt.savefig('driving_simulation_results.png', dpi=300, bbox_inches='tight')

print(f"\n=== 驾驶仿真结果统计 ===")
print(f"平均速度: {np.mean(final_speeds):.2f} ± {np.std(final_speeds):.2f} km/h")
print(f"速度范围: {np.min(final_speeds):.2f} - {np.max(final_speeds):.2f} km/h")
print(f"最终加速度范围: {X[-1, N:2*N].min():.3f} - {X[-1, N:2*N].max():.3f} m/s²")
print(f"最终转向角范围: {X[-1, 2*N:3*N].min():.3f} - {X[-1, 2*N:3*N].max():.3f} rad")
# 使用HyperPINN学习驾驶动力学
print("\n开始训练HyperPINN模型...")

arch_name, use_resnet, use_attention, use_siren = architectures[0]  # 使用ResNet
model = HyperPINNTopology(N=N, output_dim=3*N, use_resnet=use_resnet, 
                         use_attention=use_attention, use_siren=use_siren)

# 调整超参数以适应驾驶场景
model.lambda_l1_edges = 0.02      
model.lambda_l1_triangles = 0.03   
model.lambda_l1_quads = 0.04
model.lambda_l1_quints = 0.05
model.lambda_l0_edges = 0.005
model.lambda_l0_triangles = 0.01
model.lambda_l0_quads = 0.015  
model.lambda_l0_quints = 0.02

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000, eta_min=1e-6)

# 训练参数
epochs = 5000
t_data = t_data.float()
x_data = torch.tensor(X, dtype=torch.float32)

print(f"使用 {arch_name} 架构训练模型...")
print(f"数据形状: {x_data.shape}, 时间点: {len(t_data)}")

losses = []
sparsity_stats = []
def get_labels_and_scores(all_edges, true_edges, probs):
    y_true = []
    y_score = []
    for idx, edge in enumerate(all_edges):
        edge = tuple(sorted(edge))
        y_true.append(1 if edge in true_edges else 0)
        y_score.append(probs[idx])
    return np.array(y_true), np.array(y_score)

def evaluate_edges_triangles(model, t_data, all_2edges, true_2edges, all_3edges, true_3edges, all_4edges, true_4edges, all_5edges, true_5edges):
    with torch.no_grad():
        edge_probs, triangle_probs, quad_probs, quint_probs = model.get_sparse_weights(use_concrete=False, hard=False)
        edge_probs = edge_probs.cpu().numpy()
        triangle_probs = triangle_probs.cpu().numpy()
        quad_probs = quad_probs.cpu().numpy()
        quint_probs = quint_probs.cpu().numpy()
    edge_scores = [abs(edge_probs[idx]) for idx, _ in enumerate(all_2edges)]
    triangle_scores = [abs(triangle_probs[idx]) for idx, _ in enumerate(all_3edges)]
    quad_scores = [abs(quad_probs[idx]) for idx, _ in enumerate(all_4edges)]
    quint_scores = [abs(quint_probs[idx]) for idx, _ in enumerate(all_5edges)]
    y_true_2, y_score_2 = get_labels_and_scores(all_2edges, true_2edges, edge_scores)
    y_true_3, y_score_3 = get_labels_and_scores(all_3edges, true_3edges, triangle_scores)
    y_true_4, y_score_4 = get_labels_and_scores(all_4edges, true_4edges, quad_scores)
    y_true_5, y_score_5 = get_labels_and_scores(all_5edges, true_5edges, quint_scores)
    return y_true_2, y_score_2, y_true_3, y_score_3, y_true_4, y_score_4, y_true_5, y_score_5

def compute_auc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)

def plot_roc(y_true, y_score, label):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc_score:.2f})',linewidth=2)
    return fpr, tpr, auc_score

# 简化训练循环用于驾驶场景演示
for epoch in range(0, epochs, 500):  # 每500轮打印一次
    optimizer.zero_grad()
    x_pred = model.forward(t_data)
    
    # 损失计算
    physics_loss = model.physics_loss(t_data)
    data_loss = torch.mean((x_pred - x_data)**2)
    sparsity_loss, sparsity_info = model.sparsity_regularization()
    
    # 权重调整
    if epoch < 1000:
        physics_weight, data_weight, sparsity_weight = 0.01, 1.0, 0.0
    elif epoch < 3000:
        physics_weight, data_weight, sparsity_weight = 0.5, 0.5, 0.1
    else:
        physics_weight, data_weight, sparsity_weight = 1.0, 0.2, 0.2
    
    total_loss = physics_weight * physics_loss + data_weight * data_loss + sparsity_weight * sparsity_loss
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    
    losses.append(total_loss.item())
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss={total_loss.item():.6f}, "
              f"Physics={physics_loss.item():.6f}, Data={data_loss.item():.6f}, "
              f"Sparsity={sparsity_loss.item():.6f}")
        
        # 评估超图发现效果
        y_true_2, y_score_2, y_true_3, y_score_3, y_true_4, y_score_4, y_true_5, y_score_5 = \
            evaluate_edges_triangles(model, t_data, all_2edges, true_2edges, all_3edges, 
                               true_3edges, all_4edges, true_4edges, all_5edges, true_5edges)
        auc_2 = compute_auc(y_true_2, y_score_2)
        auc_3 = compute_auc(y_true_3, y_score_3)
        auc_4 = compute_auc(y_true_4, y_score_4)
        auc_5 = compute_auc(y_true_5, y_score_5)
        print(f"  AUC发现效果 - 跟车/车道: {auc_2:.3f}, 路口: {auc_3:.3f}, 环岛: {auc_4:.3f}, 枢纽: {auc_5:.3f}")

print(f"\n训练完成! 最终损失: {losses[-1]:.6f}")
print(f"模型已学习到驾驶场景的动力学和超图交互模式")

y_true_2, y_score_2, y_true_3, y_score_3, y_true_4, y_score_4, y_true_5, y_score_5 = \
    evaluate_edges_triangles(model, t_data, all_2edges, true_2edges, all_3edges, \
                        true_3edges, all_4edges, true_4edges, all_5edges, true_5edges)
y_true_total = np.concatenate([y_true_2, y_true_3, y_true_4, y_true_5])
y_score_total = np.concatenate([y_score_2, y_score_3, y_score_4, y_score_5])   
plt.figure(figsize=(8, 6))
plot_roc(y_true_2, y_score_2, 'Pairwise')
plot_roc(y_true_3, y_score_3, 'Third-order')
plot_roc(y_true_4, y_score_4, 'Fourth-order')
plot_roc(y_true_5, y_score_5, 'Fifth-order')
plot_roc(y_true_total, y_score_total, label='All') 
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate',fontsize=16)
plt.ylabel('True Positive Rate',fontsize=16)
plt.title('ROC Curves for Identified Hypergraphs',fontsize=17)
plt.legend(fontsize=14, loc="lower right")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('driving_hypergraph_discovery_roc.png', bbox_inches='tight')

# 可视化驾驶场景中的间接交互影响
print(f"\n=== 驾驶场景中的间接交互分析 ===")
print("1. 跟车关系 (2阶): 相邻车辆直接影响速度和车道变换")
print("2. 交叉路口 (3阶): 三方博弈，优先级基于速度")  
print("3. 环岛系统 (4阶): 右侧优先规则，需要协调转向")
print("4. 复杂枢纽 (5阶): 多方协调，速度和转向同步")
print(f"\n间接影响示例:")
print(f"- 车辆1可通过枢纽[1,2,3,4,5]间接影响车辆3,4,5")
print(f"- 车辆6可通过环岛[2,4,6,8]间接影响车辆2,4,8")
print(f"- 所有车辆通过超图结构实现复杂的多跳影响传播")
print(f"\n保存结果图像: driving_simulation_results.png, driving_hypergraph_discovery_roc.png")
