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

def roessler_hoi(t, x, EdgeList, TriangleList, QuadList, QuintList):
    m1 = len(x)
    N = m1 // 3
    xold = x[0:N]
    yold = x[N:2*N]
    zold = x[2*N:3*N]
    ar, br, cr = 0.2, 0.2, 5.7
    k,kD = 0.4,0.3

    coup_rete = np.zeros(N)
    coup_simplicial = np.zeros(N)
    coup_quads = np.zeros(N)
    coup_quints = np.zeros(N)
    for ii in range(len(EdgeList)):
        i1 = EdgeList[ii, 0] - 1
        i2 = EdgeList[ii, 1] - 1
        coup_rete[i1] += xold[i2] - xold[i1]
        coup_rete[i2] += xold[i1] - xold[i2]
    mtrianglelist, ntrianglelist = TriangleList.shape
    for ii in range(mtrianglelist):
        i1 = TriangleList[ii, 0] - 1
        i2 = TriangleList[ii, 1] - 1
        i3 = TriangleList[ii, 2] - 1
        coup_simplicial[i1] += xold[i2]**2 * xold[i3] - xold[i1]**3 + xold[i2] * xold[i3]**2 - xold[i1]**3
        coup_simplicial[i2] += xold[i1]**2 * xold[i3] - xold[i2]**3 + xold[i1] * xold[i3]**2 - xold[i2]**3
        coup_simplicial[i3] += xold[i1]**2 * xold[i2] - xold[i3]**3 + xold[i1] * xold[i2]**2 - xold[i3]**3
    # quads: 4-node hyperedges, follow same style as triangles but higher-order terms
    if QuadList is not None and len(QuadList) > 0:
        mquadlist, nquadlist = QuadList.shape
        for ii in range(mquadlist):
            i1 = QuadList[ii, 0] - 1
            i2 = QuadList[ii, 1] - 1
            i3 = QuadList[ii, 2] - 1
            i4 = QuadList[ii, 3] - 1
            coup_quads[i1] += xold[i2]**2 * xold[i3] * xold[i4] - xold[i1]**4
            coup_quads[i2] += xold[i1]**2 * xold[i3] * xold[i4] - xold[i2]**4
            coup_quads[i3] += xold[i1]**2 * xold[i2] * xold[i4] - xold[i3]**4
            coup_quads[i4] += xold[i1]**2 * xold[i2] * xold[i3] - xold[i4]**4
    # quints: 5-node hyperedges
    if QuintList is not None and len(QuintList) > 0:
        mquintlist, nquintlist = QuintList.shape
        for ii in range(mquintlist):
            i1 = QuintList[ii, 0] - 1
            i2 = QuintList[ii, 1] - 1
            i3 = QuintList[ii, 2] - 1
            i4 = QuintList[ii, 3] - 1
            i5 = QuintList[ii, 4] - 1
            coup_quints[i1] += xold[i2]**2 * xold[i3] * xold[i4] * xold[i5] - xold[i1]**5
            coup_quints[i2] += xold[i1]**2 * xold[i3] * xold[i4] * xold[i5] - xold[i2]**5
            coup_quints[i3] += xold[i1]**2 * xold[i2] * xold[i4] * xold[i5] - xold[i3]**5
            coup_quints[i4] += xold[i1]**2 * xold[i2] * xold[i3] * xold[i5] - xold[i4]**5
            coup_quints[i5] += xold[i1]**2 * xold[i2] * xold[i3] * xold[i4] - xold[i5]**5
    dxdt1 = -yold - zold + k * coup_rete + kD * coup_simplicial + kD * coup_quads + kD * coup_quints
    dxdt1 = -yold - zold + k * coup_rete + kD * coup_simplicial + + kD * coup_quads + kD * coup_quints
    dydt1 = xold + ar * yold
    dzdt1 = br + zold * (xold - cr)
    dxdt = np.concatenate((dxdt1, dydt1, dzdt1))  
    return dxdt

N = 8
EdgeList = np.array([[1, 2],[2, 3],[3, 4],[5, 6],[6, 7],[7, 8]])
TriangleList = np.array([[1, 2, 3],[2, 4, 5],[5, 6, 7],[6, 7, 8]])
# ground-truth 4-node and 5-node hyperedges (follow same style/format)
QuadList = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
QuintList = np.array([[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]])

all_2edges = list(combinations(range(1, N+1), 2))
all_3edges = list(combinations(range(1, N+1), 3))
all_4edges = list(combinations(range(1, N+1), 4))
all_5edges = list(combinations(range(1, N+1), 5))

true_2edges = set(tuple(sorted(edge)) for edge in EdgeList)
true_3edges = set(tuple(sorted(triangle)) for triangle in TriangleList)
true_4edges = set(tuple(sorted(quad)) for quad in QuadList)
true_5edges = set(tuple(sorted(quint)) for quint in QuintList)

M = 150
tmax = 20
dt = tmax / M
t_eval = np.linspace(0, tmax, M+1)
t_data = torch.linspace(0, tmax, M+1, requires_grad=True).unsqueeze(1) 
x0 = np.random.uniform(-1, 1, size=(3 * N,))
sol = solve_ivp(roessler_hoi, (0,tmax), x0, t_eval=t_eval, args=(EdgeList, TriangleList, QuadList, QuintList))
X = sol.y.T 
nt = len(t_eval)
dxdt = np.array([roessler_hoi(t, sol.y[:, i], EdgeList, TriangleList, QuadList, QuintList) for i, t in enumerate(sol.t)])

x_data = torch.tensor(X, dtype=torch.float64) 

architectures = [("ResNet", True, False),("Attention", False, True)]    
arch_name, use_resnet, use_attention = architectures[0]
model = HyperPINNTopology(N=N, output_dim=3*N, use_resnet=use_resnet, use_attention=use_attention)
model.lambda_l1_edges = 0.03      
model.lambda_l1_triangles = 0.05   
model.lambda_l0_edges = 0.01
model.lambda_l0_triangles = 0.02
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

def evaluate_edges_triangles(model, t_data, all_2edges, true_2edges, all_3edges, true_3edges, all_4edges, true_4edges, all_5edges, true_5edges):
    with torch.no_grad():
        edge_probs, triangle_probs, quad_probs, quint_probs = model.get_sparse_weights(t_data, use_concrete=False, hard=False)
        # Take the last time step for evaluation
        edge_probs = edge_probs[-1].cpu().numpy()
        triangle_probs = triangle_probs[-1].cpu().numpy()
        quad_probs = quad_probs[-1].cpu().numpy()
        quint_probs = quint_probs[-1].cpu().numpy()
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
         print(f"  L1 edges: {sparsity_info['l1_edges']:.2f}, L1 triangles: {sparsity_info['l1_triangles']:.2f}")
         y_true_2, y_score_2, y_true_3, y_score_3, y_true_4, y_score_4, y_true_5, y_score_5 = \
             evaluate_edges_triangles(model, t_data, all_2edges, true_2edges, all_3edges, true_3edges, all_4edges, true_4edges, all_5edges, true_5edges)
         auc_2 = compute_auc(y_true_2, y_score_2)
         auc_3 = compute_auc(y_true_3, y_score_3)
         auc_4 = compute_auc(y_true_4, y_score_4)
         auc_5 = compute_auc(y_true_5, y_score_5)
         print(f"  AUC (2-edges): {auc_2:.4f}, AUC (3-edges): {auc_3:.4f}, AUC (4-edges): {auc_4:.4f}, AUC (5-edges): {auc_5:.4f}")

y_true_2, y_score_2, y_true_3, y_score_3, y_true_4, y_score_4, y_true_5, y_score_5 = \
    evaluate_edges_triangles(model, t_data, all_2edges, true_2edges, all_3edges, true_3edges, all_4edges, true_4edges, all_5edges, true_5edges)
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
plt.savefig('roc_curves.png', bbox_inches='tight')
