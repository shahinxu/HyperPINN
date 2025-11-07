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

class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / in_features) / omega_0, 
                                           np.sqrt(6 / in_features) / omega_0)
    
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class HyperPINNTopology(nn.Module):
    def __init__(self, N, output_dim, hidden_dim=64, num_layers=4, use_resnet=True, use_attention=False, use_siren=False):
        super().__init__() 
        self.N = N  # Number of nodes
        self.use_resnet = use_resnet
        self.use_attention = use_attention
        self.use_siren = use_siren
        input_dim = 1

        if use_attention:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.ff = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, output_dim)
        elif use_resnet:
            self.input_layer = nn.Linear(input_dim, hidden_dim)
            self.res_blocks = nn.ModuleList()
            for _ in range(num_layers - 2):
                self.res_blocks.append(ResidualBlock(hidden_dim))
            self.output_layer = nn.Linear(hidden_dim, output_dim)
        elif use_siren:
            self.siren_layers = nn.ModuleList()
            self.siren_layers.append(SirenLayer(input_dim, hidden_dim, is_first=True))
            for _ in range(num_layers - 2):
                self.siren_layers.append(SirenLayer(hidden_dim, hidden_dim))
            self.output_layer = nn.Linear(hidden_dim, output_dim)
        else:
            raise ValueError("Specify one of: use_resnet=True, use_attention=True, or use_siren=True")
        
        num_edges = N * (N-1)//2
        num_triangles = N * (N-1) * (N-2) // 6
        num_quads = N * (N-1) * (N-2) * (N-3) // 24
        num_quints = N * (N-1) * (N-2) * (N-3) * (N-4) // 120
        
        # Static graph parameters (replace dynamic hypergraph)
        self.edge_weights = nn.Parameter(torch.randn(num_edges) * 0.1 - 2.0)  
        self.triangle_weights = nn.Parameter(torch.randn(num_triangles) * 0.1 - 3.0)
        self.quad_weights = nn.Parameter(torch.randn(num_quads) * 0.1 - 3.5)
        self.quint_weights = nn.Parameter(torch.randn(num_quints) * 0.1 - 4.0)
        
        self.lambda_l1_edges = 0.01      
        self.lambda_l1_triangles = 0.01 
        self.lambda_l1_quads = 0.01
        self.lambda_l1_quints = 0.01
        self.lambda_l0_edges = 0.001     
        self.lambda_l0_triangles = 0.001 
        self.lambda_l0_quads = 0.001
        self.lambda_l0_quints = 0.001
        self.temperature = 1.0           
        self.edge_indices = list(combinations(range(N), 2))
        self.triangle_indices = list(combinations(range(N), 3))
        self.quad_indices = list(combinations(range(N), 4))
        self.quint_indices = list(combinations(range(N), 5))

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
        elif self.use_siren:
            h = t
            for layer in self.siren_layers:
                h = layer(h)
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
            quad_probs = self.concrete_binary_gates(self.quad_weights, self.temperature, hard)
            quint_probs = self.concrete_binary_gates(self.quint_weights, self.temperature, hard)
        else:
            edge_probs = torch.sigmoid(self.edge_weights)
            triangle_probs = torch.sigmoid(self.triangle_weights)
            quad_probs = torch.sigmoid(self.quad_weights)
            quint_probs = torch.sigmoid(self.quint_weights)
            if hard:
                edge_probs = (edge_probs > 0.5).float()
                triangle_probs = (triangle_probs > 0.5).float()
                quad_probs = (quad_probs > 0.5).float()
                quint_probs = (quint_probs > 0.5).float()

        return edge_probs, triangle_probs, quad_probs, quint_probs
    
    def sparsity_regularization(self):
        edge_probs = torch.sigmoid(self.edge_weights)
        triangle_probs = torch.sigmoid(self.triangle_weights)
        quad_probs = torch.sigmoid(self.quad_weights)
        quint_probs = torch.sigmoid(self.quint_weights)

        l1_edges = torch.sum(edge_probs)
        l1_triangles = torch.sum(triangle_probs)
        l1_quads = torch.sum(quad_probs)
        l1_quints = torch.sum(quint_probs)

        l0_edges = torch.sum(edge_probs * (1 - edge_probs) * 4)
        l0_triangles = torch.sum(triangle_probs * (1 - triangle_probs) * 4)
        l0_quads = torch.sum(quad_probs * (1 - quad_probs) * 4)
        l0_quints = torch.sum(quint_probs * (1 - quint_probs) * 4)

        sparsity_loss = (
            self.lambda_l1_edges * l1_edges + self.lambda_l1_triangles * l1_triangles +
            self.lambda_l1_quads * l1_quads + self.lambda_l1_quints * l1_quints +
            self.lambda_l0_edges * l0_edges + self.lambda_l0_triangles * l0_triangles +
            self.lambda_l0_quads * l0_quads + self.lambda_l0_quints * l0_quints
        )

        return sparsity_loss, {
            'l1_edges': l1_edges.item(), 'l1_triangles': l1_triangles.item(),
            'l1_quads': l1_quads.item(), 'l1_quints': l1_quints.item(),
            'l0_edges': l0_edges.item(), 'l0_triangles': l0_triangles.item(),
            'l0_quads': l0_quads.item(), 'l0_quints': l0_quints.item()
        }
     
    def physics_loss(self,t):
        x_pred = self.forward(t)
        dx_dt_pred = torch.zeros_like(x_pred)
        for i in range(x_pred.shape[1]):
            dx_dt_pred[:, i] = torch.autograd.grad(x_pred[:, i].sum(), t, create_graph=True, retain_graph=True)[0].squeeze()

        N = self.N
        # Expecting output format: [v (N), pos_x (N), pos_y (N), steering (N)]
        v = x_pred[:, 0:N]
        pos_x = x_pred[:, N:2*N]
        pos_y = x_pred[:, 2*N:3*N]
        steering = x_pred[:, 3*N:4*N]

        # Physical / model parameters (match Driving.py defaults)
        v_desired = 30.0
        tau_accel = 2.0
        tau_steer = 1.5
        k_follow = 0.3
        k_lane = 0.2
        k_intersection = 0.4
        k_roundabout = 0.5
        k_complex = 0.6

        # Initialize coupling terms
        coupling_follow = torch.zeros_like(v)
        coupling_lane = torch.zeros_like(v)
        coupling_intersection = torch.zeros_like(v)
        coupling_roundabout = torch.zeros_like(v)
        coupling_complex = torch.zeros_like(v)

        # Static learned weights
        edge_probs = torch.sigmoid(self.edge_weights)
        triangle_probs = torch.sigmoid(self.triangle_weights)
        quad_probs = torch.sigmoid(self.quad_weights)
        quint_probs = torch.sigmoid(self.quint_weights)

        # Pairwise contributions
        for idx, (i, j) in enumerate(self.edge_indices):
            w = edge_probs[idx]

            # follow model: front car affects rear car asymmetrically (match Driving.py)
            diff = v[:, i] - v[:, j]
            mask = (diff > 0).float()  # 1 if i is faster than j
            # when i is front (diff>0): j gets +0.5*diff, i gets -0.1*diff
            # else (j front): i gets +0.5*( -diff ), j gets -0.1*( -diff )
            coupling_follow[:, j] += w * (mask * (0.5 * diff) + (1.0 - mask) * (0.1 * diff))
            coupling_follow[:, i] += w * (mask * (-0.1 * diff) + (1.0 - mask) * (-0.5 * diff))

            # lane-change influence via partner steering (same coefficient)
            coupling_lane[:, i] += w * torch.sin(steering[:, j]) * 0.3
            coupling_lane[:, j] += w * torch.sin(steering[:, i]) * 0.3

        # Triangle interactions (intersection-like)
        for idx, (i, j, k) in enumerate(self.triangle_indices):
            w = triangle_probs[idx]
            # priority: highest speed gets priority (non-differentiable like Driving.py)
            speeds = torch.stack([v[:, i], v[:, j], v[:, k]], dim=1)  # (B,3)
            max_idx = torch.argmax(speeds, dim=1)  # per-batch index 0/1/2
            nodes = [i, j, k]
            # apply same per-sample rules as Driving.py
            B = v.shape[0]
            for b in range(B):
                mi = int(max_idx[b].item())
                priority_car = nodes[mi]
                for node in nodes:
                    if node != priority_car:
                        coupling_intersection[b, node] += -0.4 * (v[b, priority_car] - v[b, node]) * w
                    else:
                        others = [n for n in nodes if n != node]
                        coupling_intersection[b, node] += 0.2 * w * torch.mean(torch.stack([v[b, n] for n in others]))

        # Quad interactions (roundabout-like)
        for idx, (i, j, k, l) in enumerate(self.quad_indices):
            w = quad_probs[idx]
            nodes = [i, j, k, l]
            for node in nodes:
                others = [n for n in nodes if n != node]
                right_priority = sum([1 for n in others if n < node])
                coupling_roundabout[:, node] += w * (0.3 * right_priority - 0.2 * len(others))
                coupling_roundabout[:, node] += w * 0.1 * torch.sin(steering[:, node] + 3.14159/4)

        # Quint interactions (complex hubs)
        for idx, (i, j, k, l, m) in enumerate(self.quint_indices):
            w = quint_probs[idx]
            nodes = [i, j, k, l, m]
            for node in nodes:
                others = [n for n in nodes if n != node]
                avg_speed = torch.mean(torch.stack([v[:, n] for n in others], dim=1), dim=1)
                avg_steer = torch.mean(torch.stack([steering[:, n] for n in others], dim=1), dim=1)
                coupling_complex[:, node] += w * 0.2 * (avg_speed - v[:, node])
                coupling_complex[:, node] += w * 0.1 * torch.sin(avg_steer - steering[:, node])

        # Expected derivatives following Driving.py formulas
        dvdt_expected = (1.0 / tau_accel) * (v_desired - v + 
                                             k_follow * coupling_follow +
                                             k_lane * coupling_lane +
                                             k_intersection * coupling_intersection +
                                             k_roundabout * coupling_roundabout)

        dxdt_expected = v * torch.cos(steering)
        dydt_expected = v * torch.sin(steering)
        dsdt_expected = -steering / tau_steer + 0.1 * (v - v_desired) + k_complex * coupling_complex

        expected = torch.cat([dvdt_expected, dxdt_expected, dydt_expected, dsdt_expected], dim=1)

        loss = torch.mean((dx_dt_pred - expected) ** 2)
        return loss
