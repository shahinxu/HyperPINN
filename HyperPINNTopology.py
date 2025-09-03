import torch
from torch import nn
from torch import optim as optim
import numpy as np
from itertools import combinations

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
    def __init__(self, N, output_dim, hidden_dim=64, num_layers=4, use_resnet=True, use_attention=False):
        super().__init__() 
        self.N = N  # Number of nodes
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
        
        # For pairwise and third-order interactions 
        num_edges = N * (N-1)//2
        num_triangles = N * (N-1) * (N-2) // 6
        
        self.edge_weights = nn.Parameter(torch.randn(num_edges) * 0.1 - 2.0)  
        self.triangle_weights = nn.Parameter(torch.randn(num_triangles) * 0.1 - 3.0)       
        self.lambda_l1_edges = 0.01      
        self.lambda_l1_triangles = 0.01 
        self.lambda_l0_edges = 0.001     
        self.lambda_l0_triangles = 0.001 
        self.temperature = 1.0           
        self.edge_indices = list(combinations(range(N), 2))
        self.triangle_indices = list(combinations(range(N), 3))

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
    
    def get_sparse_weights(self, use_concrete=True, hard=False):
        if use_concrete:
            edge_probs = self.concrete_binary_gates(self.edge_weights, self.temperature, hard)
            triangle_probs = self.concrete_binary_gates(self.triangle_weights, self.temperature, hard)
        else:
            edge_probs = torch.sigmoid(self.edge_weights)
            triangle_probs = torch.sigmoid(self.triangle_weights)
            if hard:
                edge_probs = (edge_probs > 0.5).float()
                triangle_probs = (triangle_probs > 0.5).float()  
        return edge_probs, triangle_probs
    
    def sparsity_regularization(self):
        edge_probs = torch.sigmoid(self.edge_weights)
        triangle_probs = torch.sigmoid(self.triangle_weights)

        l1_edges = torch.sum(edge_probs)
        l1_triangles = torch.sum(triangle_probs)
        l0_edges = torch.sum(edge_probs * (1 - edge_probs) * 4)  
        l0_triangles = torch.sum(triangle_probs * (1 - triangle_probs) * 4)
        sparsity_loss = (self.lambda_l1_edges * l1_edges + self.lambda_l1_triangles * l1_triangles +
                        self.lambda_l0_edges * l0_edges + self.lambda_l0_triangles * l0_triangles)
        
        return sparsity_loss, {'l1_edges': l1_edges.item(),'l1_triangles': l1_triangles.item(),
                               'l0_edges': l0_edges.item(),'l0_triangles': l0_triangles.item()}
     
    def physics_loss(self,t):
        x_pred = self.forward(t)
        dx_dt_pred = torch.zeros_like(x_pred)
        for i in range(x_pred.shape[1]):
            dx_dt_pred[:, i] = torch.autograd.grad(x_pred[:, i].sum(), t, create_graph=True, retain_graph=True)[0].squeeze()
            
        N = self.N
        xold = x_pred[:,0:N]
        yold = x_pred[:,N:2*N]
        zold = x_pred[:,2*N:3*N]
        ar, br, cr = 0.2, 0.2, 5.7
        k,kD = 0.4,0.3
        coup_rete = torch.zeros_like(xold)
        coup_triangular = torch.zeros_like(xold)
        adj_matrix = torch.sigmoid(self.edge_weights)
        triangle_weights_sigmoid = torch.sigmoid(self.triangle_weights)
        
        for idx, (i, j) in enumerate(self.edge_indices):
            weight =  adj_matrix[idx]
            coup_rete[:, i] += weight * (xold[:, j] - xold[:, i])
            coup_rete[:, j] += weight * (xold[:, i] - xold[:, j])
        for idx, (i, j, k) in enumerate(self.triangle_indices):
            weight = triangle_weights_sigmoid[idx]
            term_i = weight * (xold[:, j]**2 * xold[:, k] - xold[:, i]**3 + 
                             xold[:, j] * xold[:, k]**2 - xold[:, i]**3)
            term_j = weight * (xold[:, i]**2 * xold[:, k] - xold[:, j]**3 + 
                             xold[:, i] * xold[:, k]**2 - xold[:, j]**3)
            term_k = weight * (xold[:, i]**2 * xold[:, j] - xold[:, k]**3 + 
                             xold[:, i] * xold[:, j]**2 - xold[:, k]**3)   
            coup_triangular[:, i] += term_i
            coup_triangular[:, j] += term_j
            coup_triangular[:, k] += term_k
        
        term1 = -yold - zold + k * coup_rete + kD * coup_triangular
        term2 = xold + ar * yold
        term3 = br + zold * (xold - cr)
        term  = torch.cat([term1, term2, term3],dim=1)  
        return torch.mean((dx_dt_pred - term) ** 2) 
