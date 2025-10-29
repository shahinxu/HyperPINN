"""
扩展的Rössler振荡器系统 - 支持高阶交互 (order 3-8)

使用方法：
1. 修改 order 变量 (第14行) 来设置最大交互阶数：
   - order = 3: 使用边和三角形 (原始版本)
   - order = 4: 添加四面体交互
   - order = 5: 添加5-单纯形交互
   - order = 6: 添加6-单纯形交互
   - order = 7: 添加7-单纯形交互  
   - order = 8: 添加8-单纯形交互 (全连接)

2. 运行脚本将自动使用相应阶数的图数据结构
3. 结果将保存为 'roc_curves_order_{order}.png'
"""

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

def roessler_hoi_extended(t, x, simplex_lists, max_order):
    """
    扩展的Rössler系统，支持任意阶数的高阶交互
    
    Args:
        t: 时间
        x: 状态向量
        simplex_lists: 字典，键为阶数，值为对应的单纯形列表
        max_order: 最大交互阶数
    """
    m1 = len(x)
    N = m1 // 3
    xold = x[0:N]
    yold = x[N:2*N]
    zold = x[2*N:3*N]
    ar, br, cr = 0.2, 0.2, 5.7
    k, kD = 0.4, 0.3

    coup_total = np.zeros(N)
    
    # 处理每个阶数的交互
    for curr_order in range(2, max_order + 1):
        if curr_order not in simplex_lists or len(simplex_lists[curr_order]) == 0:
            continue
            
        # 耦合强度随阶数递减
        coupling_strength = k if curr_order == 2 else kD * (0.7 ** (curr_order - 3))
        
        if curr_order == 2:
            # 二阶交互 (边)
            simplex_array = simplex_lists[curr_order]
            for ii in range(len(simplex_array)):
                i1, i2 = simplex_array[ii, 0] - 1, simplex_array[ii, 1] - 1
                coup_total[i1] += coupling_strength * (xold[i2] - xold[i1])
                coup_total[i2] += coupling_strength * (xold[i1] - xold[i2])
        
        elif curr_order == 3:
            # 三阶交互 (三角形)
            simplex_array = simplex_lists[curr_order]
            for ii in range(len(simplex_array)):
                i1, i2, i3 = simplex_array[ii, 0] - 1, simplex_array[ii, 1] - 1, simplex_array[ii, 2] - 1
                coup_total[i1] += coupling_strength * (xold[i2]**2 * xold[i3] - xold[i1]**3 + 
                                                     xold[i2] * xold[i3]**2 - xold[i1]**3)
                coup_total[i2] += coupling_strength * (xold[i1]**2 * xold[i3] - xold[i2]**3 + 
                                                     xold[i1] * xold[i3]**2 - xold[i2]**3)
                coup_total[i3] += coupling_strength * (xold[i1]**2 * xold[i2] - xold[i3]**3 + 
                                                     xold[i1] * xold[i2]**2 - xold[i3]**3)
        
        elif curr_order == 4:
            # 四阶交互 (四面体)
            simplex_array = simplex_lists[curr_order]
            for ii in range(len(simplex_array)):
                indices = [simplex_array[ii, j] - 1 for j in range(4)]
                for j, node_idx in enumerate(indices):
                    other_indices = [indices[k] for k in range(4) if k != j]
                    interaction_term = coupling_strength * (xold[other_indices[0]] * xold[other_indices[1]] * 
                                                          xold[other_indices[2]] - xold[node_idx]**3)
                    coup_total[node_idx] += interaction_term
        
        else:
            # 高阶交互 (5阶及以上)
            simplex_array = simplex_lists[curr_order]
            for ii in range(len(simplex_array)):
                indices = [simplex_array[ii, j] - 1 for j in range(curr_order)]
                for j, node_idx in enumerate(indices):
                    other_indices = [indices[k] for k in range(curr_order) if k != j]
                    # 所有其他节点的乘积减去自交互
                    interaction_term = coupling_strength
                    for other_idx in other_indices:
                        interaction_term *= xold[other_idx]
                    interaction_term -= coupling_strength * xold[node_idx]**(curr_order-1)
                    coup_total[node_idx] += interaction_term

    dxdt1 = -yold - zold + coup_total
    dydt1 = xold + ar * yold
    dzdt1 = br + zold * (xold - cr)
    dxdt = np.concatenate((dxdt1, dydt1, dzdt1))  
    return dxdt

# 为了向后兼容，保留原函数作为包装器
def roessler_hoi(t, x, EdgeList, TriangleList):
    """原始函数的向后兼容包装器"""
    simplex_lists = {2: EdgeList, 3: TriangleList}
    return roessler_hoi_extended(t, x, simplex_lists, 3)

N = 8
order = 3  # 设置交互的最大阶数，可以是 3, 4, 5, 6, 7, 8

# 定义不同阶数的图数据结构
EdgeList = np.array([[1, 2],[2, 3],[3, 4],[5, 6],[6, 7],[7, 8]])  # 2阶 (边)
TriangleList = np.array([[1, 2, 3],[2, 4, 5],[5, 6, 7],[6, 7, 8]])  # 3阶 (三角形)
QuadList = np.array([[1, 2, 3, 4],[5, 6, 7, 8]])  # 4阶 (四面体)
PentaList = np.array([[1, 2, 3, 4, 5]])  # 5阶 (5-单纯形)
HexaList = np.array([[2, 3, 4, 5, 6, 7]])  # 6阶 (6-单纯形)
SeptaList = np.array([[1, 2, 3, 4, 5, 6, 7]])  # 7阶 (7-单纯形)
OctaList = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])  # 8阶 (8-单纯形)

# 根据 order 参数选择要使用的数据结构
simplex_lists = {}
if order >= 2:
    simplex_lists[2] = EdgeList
if order >= 3:
    simplex_lists[3] = TriangleList
if order >= 4:
    simplex_lists[4] = QuadList
if order >= 5:
    simplex_lists[5] = PentaList
if order >= 6:
    simplex_lists[6] = HexaList
if order >= 7:
    simplex_lists[7] = SeptaList
if order >= 8:
    simplex_lists[8] = OctaList

# 生成所有可能的简单形和真实的简单形集合
all_simplexes = {}
true_simplex_sets = {}
for k in range(2, order + 1):
    all_simplexes[k] = list(combinations(range(1, N+1), k))
    if k in simplex_lists:
        true_simplex_sets[k] = set(tuple(sorted(simplex)) for simplex in simplex_lists[k])
    else:
        true_simplex_sets[k] = set()

# 为了向后兼容，保留原有变量名
all_2edges = all_simplexes.get(2, [])
all_3edges = all_simplexes.get(3, [])
true_2edges = true_simplex_sets.get(2, set())
true_3edges = true_simplex_sets.get(3, set())

M = 150
tmax = 20
dt = tmax / M
t_eval = np.linspace(0, tmax, M+1)
t_data = torch.linspace(0, tmax, M+1, requires_grad=True).unsqueeze(1) 
x0 = np.random.uniform(-1, 1, size=(3 * N,))
sol = solve_ivp(roessler_hoi_extended, (0,tmax), x0, t_eval=t_eval, args=(simplex_lists, order))
X = sol.y.T 
nt = len(t_eval)
dxdt = np.array([roessler_hoi_extended(t, sol.y[:, i], simplex_lists, order) for i, t in enumerate(sol.t)])

## Add noise
#noise_level = 0.1 * np.std(sol.y) 
#X += noise_level * np.random.randn(*X.shape)
x_data = torch.tensor(X, dtype=torch.float64) 

architectures = [("ResNet", True, False),("Attention", False, True)]    
arch_name, use_resnet, use_attention = architectures[0]
model = HyperPINNTopology(N=N, output_dim=3*N, use_resnet=use_resnet, use_attention=use_attention, max_order=order)

# 根据 order 设置正则化参数
for curr_order in range(2, order + 1):
    if curr_order == 2:
        model.lambda_l1[curr_order] = 0.03
        model.lambda_l0[curr_order] = 0.01
    elif curr_order == 3:
        model.lambda_l1[curr_order] = 0.05
        model.lambda_l0[curr_order] = 0.02
    else:
        # 高阶交互使用递增的正则化参数
        factor = 1.5 ** (curr_order - 3)
        model.lambda_l1[curr_order] = 0.05 * factor
        model.lambda_l0[curr_order] = 0.02 * factor

# 为了向后兼容，保留原有变量名
if hasattr(model, 'lambda_l1_edges'):
    model.lambda_l1_edges = model.lambda_l1.get(2, 0.03)
if hasattr(model, 'lambda_l1_triangles'):
    model.lambda_l1_triangles = model.lambda_l1.get(3, 0.05)
if hasattr(model, 'lambda_l0_edges'):
    model.lambda_l0_edges = model.lambda_l0.get(2, 0.01)
if hasattr(model, 'lambda_l0_triangles'):
    model.lambda_l0_triangles = model.lambda_l0.get(3, 0.02)
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
losses = []
sparsity_stats = []
t_data = t_data.float()
x_data = x_data.float()

epochs = 14000
stage1_epochs = 2500   
stage2_epochs = 10000 
adaptive_weights = True
best_loss = float('inf')
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
def get_labels_and_scores(all_edges, true_edges, probs):
    y_true = []
    y_score = []
    for idx, edge in enumerate(all_edges):
        edge = tuple(sorted(edge))
        y_true.append(1 if edge in true_edges else 0)
        y_score.append(probs[idx])
    return np.array(y_true), np.array(y_score)

def evaluate_multi_order_interactions(model, t_data, all_simplexes, true_simplex_sets, max_order):
    """评估多阶交互的函数"""
    with torch.no_grad():
        _, _, weights_by_order = model.get_sparse_weights(t_data, use_concrete=False, hard=False)
    
    y_true_all = []
    y_score_all = []
    results_by_order = {}
    
    for curr_order in range(2, max_order + 1):
        if curr_order not in all_simplexes or curr_order not in weights_by_order:
            continue
            
        y_true = []
        y_score = []
        all_curr_simplexes = all_simplexes[curr_order]
        true_set = true_simplex_sets.get(curr_order, set())
        probs = weights_by_order[curr_order][-1].cpu().numpy()  # 最后一个时间步
        
        for idx, simplex in enumerate(all_curr_simplexes):
            simplex_tuple = tuple(sorted(simplex))
            y_true.append(1 if simplex_tuple in true_set else 0)
            y_score.append(abs(probs[idx]) if idx < len(probs) else 0.0)
        
        results_by_order[curr_order] = (np.array(y_true), np.array(y_score))
        y_true_all.extend(y_true)
        y_score_all.extend(y_score)
    
    return np.array(y_true_all), np.array(y_score_all), results_by_order

def evaluate_edges_triangles(model, t_data, all_2edges, true_2edges, all_3edges, true_3edges):
    """向后兼容的评估函数"""
    all_simplexes = {2: all_2edges, 3: all_3edges}
    true_simplex_sets = {2: true_2edges, 3: true_3edges}
    _, _, results_by_order = evaluate_multi_order_interactions(model, t_data, all_simplexes, true_simplex_sets, 3)
    
    y_true_2, y_score_2 = results_by_order.get(2, (np.array([]), np.array([])))
    y_true_3, y_score_3 = results_by_order.get(3, (np.array([]), np.array([])))
    
    return y_true_2, y_score_2, y_true_3, y_score_3

def compute_auc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)

def plot_roc(y_true, y_score, label):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc_score:.2f})',linewidth=2)
    return fpr, tpr, auc_score

for epoch in range(epochs):
    optimizer.zero_grad()    
    x_pred = model.forward(t_data)
    physics_loss = model.physics_loss(t_data)    
    data_loss = torch.mean((x_pred - x_data)**2)
    sparsity_loss, sparsity_info = model.sparsity_regularization(t_data)        
    if adaptive_weights and epoch > 500:
        sparsity_weight = max(0.1, 1.0 * (0.99 ** (epoch - 500)))
    else:
        sparsity_weight = 1.0
       
    if epoch < stage1_epochs:
        physics_weight = 0.01    
        data_weight = 1.0        
        sparsity_weight = 0.0    
        print_prefix = "Stage 1 (Data Fitting)"      
    elif epoch < stage2_epochs:
        progress = (epoch - stage1_epochs) / (stage2_epochs - stage1_epochs)
        physics_weight = 0.01 + 0.99 * progress  
        data_weight = 1.0 - 0.8 * progress       
        sparsity_weight = 0.0    
        print_prefix = "Stage 2 (Physics Learning)"      
    else:
        progress = min(1.0, (epoch - stage2_epochs) / (epochs - stage2_epochs))
        physics_weight = 1.0    
        data_weight = 0.2        
        sparsity_weight = 0.1 * progress  
        if hasattr(model, 'temperature'):
            model.temperature = max(0.5, 1.0 * (0.995 ** ((epoch - stage2_epochs) // 100)))
           
    total_loss = physics_weight * physics_loss + data_weight * data_loss + sparsity_weight * sparsity_loss       
    total_loss.backward() 
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   
    optimizer.step()
    scheduler.step()     
    losses.append(total_loss.item())
    sparsity_stats.append(sparsity_info)
        
    if epoch % 500 == 0:
         print(f"Epoch {epoch}, Loss: {total_loss.item():.6f}")
         print(f"  Physics: {physics_loss.item():.6f}, Data: {data_loss.item():.6f}")
         print(f"  Sparsity: {sparsity_loss.item():.6f}")
         
         # 打印所有阶数的稀疏性信息
         for curr_order in range(2, order + 1):
             l1_key = f'l1_order_{curr_order}'
             if l1_key in sparsity_info:
                 print(f"  L1 order {curr_order}: {sparsity_info[l1_key]:.2f}")
         
         # 评估所有阶数的AUC
         y_true_all, y_score_all, results_by_order = evaluate_multi_order_interactions(
             model, t_data, all_simplexes, true_simplex_sets, order)
         
         total_auc = compute_auc(y_true_all, y_score_all) if len(y_true_all) > 0 else 0.0
         print(f"  Total AUC: {total_auc:.4f}")
         
         for curr_order, (y_true, y_score) in results_by_order.items():
             if len(y_true) > 0:
                 auc_order = compute_auc(y_true, y_score)
                 print(f"  AUC (order {curr_order}): {auc_order:.4f}")

# 最终评估所有阶数的交互
print(f"\n=== 最终评估 (Order = {order}) ===")
y_true_all, y_score_all, results_by_order = evaluate_multi_order_interactions(
    model, t_data, all_simplexes, true_simplex_sets, order)

plt.figure(figsize=(10, 6))

# 绘制每个阶数的ROC曲线
order_names = {2: 'Pairwise', 3: 'Third-order', 4: 'Fourth-order', 
               5: 'Fifth-order', 6: 'Sixth-order', 7: 'Seventh-order', 8: 'Eighth-order'}

for curr_order, (y_true, y_score) in results_by_order.items():
    if len(y_true) > 0 and len(np.unique(y_true)) >= 2:  # 确保有两个类别
        order_name = order_names.get(curr_order, f'Order-{curr_order}')
        plot_roc(y_true, y_score, order_name)

# 绘制总体ROC曲线
if len(y_true_all) > 0 and len(np.unique(y_true_all)) >= 2:
    plot_roc(y_true_all, y_score_all, f'All Orders (1-{order})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title(f'ROC Curves for Identified Hypergraphs (Max Order = {order})', fontsize=17)
plt.legend(fontsize=12, loc="lower right")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(f'roc_curves_order_{order}.png', bbox_inches='tight')

# 打印最终统计信息
print(f"\n=== 最终统计 ===")
for curr_order, (y_true, y_score) in results_by_order.items():
    if len(y_true) > 0:
        auc_score = compute_auc(y_true, y_score)
        true_count = np.sum(y_true)
        total_count = len(y_true)
        order_name = order_names.get(curr_order, f'Order-{curr_order}')
        print(f"{order_name}: AUC = {auc_score:.4f}, True/Total = {true_count}/{total_count}")

if len(y_true_all) > 0:
    total_auc = compute_auc(y_true_all, y_score_all)
    print(f"Overall AUC: {total_auc:.4f}")

plt.show()
