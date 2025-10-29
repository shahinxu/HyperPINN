import torch
from torch import nn
from torch import optim as optim
import numpy as np
from itertools import combinations
from math import comb

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )  
    def forward(self, x):
        return x + self.net(x)

class HyperPINNTopology(nn.Module):
    def __init__(self, N, output_dim, hidden_dim=64, num_layers=4, use_resnet=True, use_attention=False, max_order=3):
        super().__init__() 
        self.N = N  # Number of nodes
        self.max_order = max_order  # Maximum order of interactions
        self.use_resnet = use_resnet
        self.use_attention = use_attention
        input_dim = 1

        if use_attention:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.ff = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, output_dim)
        elif use_resnet:
            self.input_layer = nn.Linear(input_dim, hidden_dim)
            self.res_blocks = nn.ModuleList()
            for _ in range(num_layers - 2):
                self.res_blocks.append(ResidualBlock(hidden_dim))
            self.output_layer = nn.Linear(hidden_dim, output_dim)
        else:
            raise ValueError("Specify either use_resnet=True or use_attention=True")
        
        # Calculate number of simplexes for each order
        self.simplex_counts = {}
        self.simplex_indices = {}
        total_simplexes = 0
        
        for order in range(2, max_order + 1):
            # For order k, we have k-simplexes (e.g., order 2 = edges, order 3 = triangles)
            count = comb(N, order)
            self.simplex_counts[order] = count
            self.simplex_indices[order] = list(combinations(range(N), order))
            total_simplexes += count
        
        # Dynamic hypergraph network: input t, output weights for all orders
        self.dynamic_hypergraph = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, total_simplexes)
        )
        
        # Regularization parameters for each order
        self.lambda_l1 = {}
        self.lambda_l0 = {}
        for order in range(2, max_order + 1):
            if order == 2:
                self.lambda_l1[order] = 0.01
                self.lambda_l0[order] = 0.001
            elif order == 3:
                self.lambda_l1[order] = 0.01
                self.lambda_l0[order] = 0.001
            else:
                # Adjust regularization for higher orders
                self.lambda_l1[order] = 0.01 * (1.2 ** (order - 3))
                self.lambda_l0[order] = 0.001 * (1.2 ** (order - 3))
        
        self.temperature = 1.0
        
        # For backward compatibility
        self.edge_indices = self.simplex_indices.get(2, [])
        self.triangle_indices = self.simplex_indices.get(3, [])

    def forward(self, t):
        t = t.float()
        if t.ndim == 1:
            t = t.unsqueeze(1)
        if self.use_attention:
            h = self.input_proj(t).unsqueeze(1)
            attn_out, _ = self.attention(h, h, h)
            h = self.norm1(h + attn_out)
            ff_out = self.ff(h)
            h = self.norm2(h + ff_out)
            return self.output_proj(h.squeeze(1))   
        elif self.use_resnet:
            h = torch.tanh(self.input_layer(t))
            for block in self.res_blocks:
                h = block(h)
            return self.output_layer(h)
  
    def concrete_binary_gates(self, logits, temperature=1.0, hard=False):
        uniform = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(uniform + 1e-20) + 1e-20)
        y_soft = torch.sigmoid((logits + gumbel) / temperature)
        
        if hard:
            # Straight-through estimator
            y_hard = (y_soft > 0.5).float()
            y = y_hard - y_soft.detach() + y_soft
        else:
            y = y_soft     
        return y
    
    def get_sparse_weights(self, t, use_concrete=True, hard=False):
        raw_weights = self.dynamic_hypergraph(t)
        
        # Split weights by order
        weights_by_order = {}
        start_idx = 0
        
        for order in range(2, self.max_order + 1):
            count = self.simplex_counts[order]
            end_idx = start_idx + count
            logits = raw_weights[:, start_idx:end_idx]
            
            if use_concrete:
                probs = self.concrete_binary_gates(logits, self.temperature, hard)
            else:
                probs = torch.sigmoid(logits)
                if hard:
                    probs = (probs > 0.5).float()
            
            weights_by_order[order] = probs
            start_idx = end_idx
        
        # For backward compatibility, return edge and triangle weights
        edge_probs = weights_by_order.get(2, torch.empty(t.shape[0], 0))
        triangle_probs = weights_by_order.get(3, torch.empty(t.shape[0], 0))
        
        return edge_probs, triangle_probs, weights_by_order
    
    def sparsity_regularization(self, t):
        raw_weights = self.dynamic_hypergraph(t)
        
        # Split weights by order
        start_idx = 0
        sparsity_terms = []
        sparsity_info = {}
        
        for order in range(2, self.max_order + 1):
            count = self.simplex_counts[order]
            end_idx = start_idx + count
            logits = raw_weights[:, start_idx:end_idx]
            probs = torch.sigmoid(logits)
            
            l1_term = torch.sum(probs)
            l0_term = torch.sum(probs * (1 - probs) * 4)
            
            order_sparsity = self.lambda_l1[order] * l1_term + self.lambda_l0[order] * l0_term
            sparsity_terms.append(order_sparsity)
            
            sparsity_info[f'l1_order_{order}'] = l1_term.item()
            sparsity_info[f'l0_order_{order}'] = l0_term.item()
            
            start_idx = end_idx
        
        # Sum all sparsity terms (avoiding inplace operations)
        sparsity_loss = torch.sum(torch.stack(sparsity_terms)) if sparsity_terms else torch.tensor(0.0)
        
        # For backward compatibility
        sparsity_info['l1_edges'] = sparsity_info.get('l1_order_2', 0.0)
        sparsity_info['l1_triangles'] = sparsity_info.get('l1_order_3', 0.0)
        sparsity_info['l0_edges'] = sparsity_info.get('l0_order_2', 0.0)
        sparsity_info['l0_triangles'] = sparsity_info.get('l0_order_3', 0.0)
        
        return sparsity_loss, sparsity_info
     
    def physics_loss(self, t):
        """
        计算物理损失函数
        
        注意: 这里使用 torch.autograd.grad 计算神经网络输出对时间的梯度 (∂x_pred/∂t)，
        这与 Rossler_Oscillators.py 中的 roessler_hoi_extended 函数不同：
        - HyperPINNTopology: 自动微分 ∂x_pred/∂t (神经网络方法)  
        - roessler_hoi_extended: 解析微分方程 dx/dt = f(x,t) (数值积分方法)
        
        两种方法的物理方程实现必须完全一致，但微分计算方式不同是正常的。
        """
        x_pred = self.forward(t)
        dx_dt_pred = torch.zeros_like(x_pred)
        for i in range(x_pred.shape[1]):
            dx_dt_pred[:, i] = torch.autograd.grad(x_pred[:, i].sum(), t, create_graph=True, retain_graph=True)[0].squeeze()
            
        N = self.N
        xold = x_pred[:,0:N]
        yold = x_pred[:,N:2*N]
        zold = x_pred[:,2*N:3*N]
        ar, br, cr = 0.2, 0.2, 5.7
        k, kD = 0.4, 0.3
        
        # Initialize coupling contributions for each node (avoid inplace operations)
        coupling_contributions = []
        for node in range(xold.shape[1]):
            coupling_contributions.append([])
        
        raw_weights = self.dynamic_hypergraph(t)
        start_idx = 0
        
        # Process each order of interactions
        for order in range(2, self.max_order + 1):
            count = self.simplex_counts[order]
            end_idx = start_idx + count
            weights = torch.sigmoid(raw_weights[:, start_idx:end_idx])
            
            # Coupling strength decreases with order
            coupling_strength = k if order == 2 else kD * (0.7 ** (order - 3))
            
            if order == 2:
                # Pairwise interactions (edges)
                for idx, (i, j) in enumerate(self.simplex_indices[2]):
                    weight = weights[:, idx]
                    term_i = coupling_strength * weight * (xold[:, j] - xold[:, i])
                    term_j = coupling_strength * weight * (xold[:, i] - xold[:, j])
                    coupling_contributions[i].append(term_i)
                    coupling_contributions[j].append(term_j)
            
            elif order == 3:
                # Third-order interactions (triangles) - 与 Rossler_Oscillators.py 保持一致
                for idx, (i, j, k) in enumerate(self.simplex_indices[3]):
                    weight = weights[:, idx]
                    # 确保与 roessler_hoi_extended 中的公式完全一致
                    term_i = coupling_strength * weight * (xold[:, j]**2 * xold[:, k] - xold[:, i]**3 + 
                                                         xold[:, j] * xold[:, k]**2 - xold[:, i]**3)
                    term_j = coupling_strength * weight * (xold[:, i]**2 * xold[:, k] - xold[:, j]**3 + 
                                                         xold[:, i] * xold[:, k]**2 - xold[:, j]**3)
                    term_k = coupling_strength * weight * (xold[:, i]**2 * xold[:, j] - xold[:, k]**3 + 
                                                         xold[:, i] * xold[:, j]**2 - xold[:, k]**3)   
                    coupling_contributions[i].append(term_i)
                    coupling_contributions[j].append(term_j)
                    coupling_contributions[k].append(term_k)
            
            elif order == 4:
                # Fourth-order interactions (tetrahedra) - 与 Rossler_Oscillators.py 保持一致
                for idx, (i, j, k, l) in enumerate(self.simplex_indices[4]):
                    weight = weights[:, idx]
                    # 与 roessler_hoi_extended 中的实现完全一致
                    for node_pos, node in enumerate([i, j, k, l]):
                        other_nodes = [n for pos, n in enumerate([i, j, k, l]) if pos != node_pos]
                        interaction_term = coupling_strength * weight * (xold[:, other_nodes[0]] * xold[:, other_nodes[1]] * 
                                                                      xold[:, other_nodes[2]] - xold[:, node]**3)
                        coupling_contributions[node].append(interaction_term)
            
            else:
                # Higher-order interactions (order >= 5) - 与 Rossler_Oscillators.py 保持一致
                for idx, simplex in enumerate(self.simplex_indices[order]):
                    weight = weights[:, idx]
                    # 与 roessler_hoi_extended 中的实现完全一致
                    for node_pos, node in enumerate(simplex):
                        other_nodes = [n for pos, n in enumerate(simplex) if pos != node_pos]
                        # 所有其他节点的乘积减去自交互
                        interaction_term = weight
                        for other_node in other_nodes:
                            interaction_term = interaction_term * xold[:, other_node]
                        interaction_term = interaction_term - weight * xold[:, node]**(order-1)
                        final_term = coupling_strength * interaction_term
                        coupling_contributions[node].append(final_term)
            
            start_idx = end_idx
        
        # Sum all contributions for each node (avoiding inplace operations)
        coup_total = torch.zeros_like(xold)
        for node in range(xold.shape[1]):
            if coupling_contributions[node]:
                coup_total[:, node] = torch.sum(torch.stack(coupling_contributions[node]), dim=0)
        
        term1 = -yold - zold + coup_total
        term2 = xold + ar * yold
        term3 = br + zold * (xold - cr)
        term  = torch.cat([term1, term2, term3], dim=1)  
        return torch.mean((dx_dt_pred - term) ** 2) 
