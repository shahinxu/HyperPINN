import torch
from torch import nn
from torch import optim
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import lstsq
from sympy import symbols, Matrix, lambdify

torch.set_default_dtype(torch.float64)
class HyperPINNWeight(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,A_mask):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()  
        self.A = nn.Parameter(torch.randn(output_dim, output_dim) * 0.1)  
        self.Tri1 = nn.Parameter(torch.tensor([0.005], dtype=torch.float32))
        self.Tri2 = nn.Parameter(torch.tensor([0.001], dtype=torch.float32))
        self.register_buffer('A_mask', A_mask)  

    def get_structured_A(self):
        return self.A * self.A_mask
        
    def forward(self, t):
        x = self.activation(self.fc1(t))
        x = self.activation(self.fc2(x))    
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x,self.get_structured_A()
    
    def physics_loss(self, t):
        R_tensor = torch.tensor([0.6099, 0.6177, 0.8594, 0.8055, 0.5767, 0.1829, 0.2399], dtype=torch.float64, requires_grad=False)
        K_tensor = torch.tensor([88.7647, 3.8387, 49.5002, 17.6248, 97.8894, 71.5568, 50.5467], dtype=torch.float64, requires_grad=False)
        x_pred, A_pred = self.forward(t)
        dx_dt_pred = torch.zeros_like(x_pred)
        for i in range(x_pred.shape[1]):
            dx_dt_pred[:, i] = torch.autograd.grad(x_pred[:, i].sum(), t, create_graph=True, retain_graph=True)[0].squeeze()  
        term1 = x_pred * R_tensor * (1 - x_pred / K_tensor)      
        term2 = x_pred * (torch.matmul(A_pred, x_pred.T).T)
        coup_simplicial = torch.zeros_like(x_pred)
        coup_simplicial[:,1] += self.Tri1 * x_pred[:,2] * x_pred[:,6]
        coup_simplicial[:,3] += self.Tri2 * x_pred[:,0] * x_pred[:,5]
        term3 = x_pred * coup_simplicial  
        return torch.mean((dx_dt_pred - term1 - term2 - term3) ** 2) 
