from HyperPINNTopology import HyperPINNTopology
import torch
from torch import nn
from torch import optim as optim
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from itertools import combinations
from sklearn.metrics import roc_curve, auc


def build_simplex_sets(N, order, simplex_lists):
    """Build all possible simplexes and the set of ground-truth simplexes.

    Returns:
        all_simplexes: dict mapping k -> list of tuples (1-based indices)
        true_simplex_sets: dict mapping k -> set of tuples (1-based indices)
    """
    all_simplexes = {}
    true_simplex_sets = {}
    for k in range(2, order + 1):
        all_simplexes[k] = list(combinations(range(1, N + 1), k))
        if k in simplex_lists:
            # ensure each simplex is a sorted tuple
            true_simplex_sets[k] = set(tuple(sorted(tuple(s))) for s in simplex_lists[k])
        else:
            true_simplex_sets[k] = set()
    return all_simplexes, true_simplex_sets

def roessler_hoi_extended(t, x, simplex_lists, max_order):
    m1 = len(x)
    N = m1 // 3
    xold = x[0:N]
    yold = x[N:2*N]
    zold = x[2*N:3*N]
    ar, br, cr = 0.2, 0.2, 5.7
    k, kD = 0.4, 0.3

    coup_total = np.zeros(N)
    
    for curr_order in range(2, max_order + 1):
        if curr_order not in simplex_lists or len(simplex_lists[curr_order]) == 0:
            continue
            
        coupling_strength = k if curr_order == 2 else kD * (0.7 ** (curr_order - 3))
        
        if curr_order == 2:
            simplex_array = simplex_lists[curr_order]
            for ii in range(len(simplex_array)):
                i1, i2 = simplex_array[ii, 0] - 1, simplex_array[ii, 1] - 1
                coup_total[i1] += coupling_strength * (xold[i2] - xold[i1])
                coup_total[i2] += coupling_strength * (xold[i1] - xold[i2])
        
        elif curr_order == 3:
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
            simplex_array = simplex_lists[curr_order]
            for ii in range(len(simplex_array)):
                indices = [simplex_array[ii, j] - 1 for j in range(4)]
                for j, node_idx in enumerate(indices):
                    other_indices = [indices[k] for k in range(4) if k != j]
                    interaction_term = coupling_strength * (xold[other_indices[0]] * xold[other_indices[1]] * 
                                                          xold[other_indices[2]] - xold[node_idx]**3)
                    coup_total[node_idx] += interaction_term
        
        else:
            simplex_array = simplex_lists[curr_order]
            for ii in range(len(simplex_array)):
                indices = [simplex_array[ii, j] - 1 for j in range(curr_order)]
                for j, node_idx in enumerate(indices):
                    other_indices = [indices[k] for k in range(curr_order) if k != j]
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


def default_simplex_lists(N, order):
    """Return a dictionary of default simplex lists (1-based indices) for small demo networks.

    This function encapsulates the hard-coded example simplexes used previously. It
    keeps 1-based indexing to match the rest of the file. For larger N or custom
    experiments, pass a custom simplex_lists dict to RosslerExperiment.
    """
    EdgeList = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8]])
    TriangleList = np.array([[1, 2, 3], [2, 4, 5], [5, 6, 7], [6, 7, 8]])
    QuadList = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    PentaList = np.array([[1, 2, 3, 4, 5]])
    HexaList = np.array([[2, 3, 4, 5, 6, 7]])
    SeptaList = np.array([[1, 2, 3, 4, 5, 6, 7]])
    OctaList = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])

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

    return simplex_lists


# Note: `N`, `order`, and construction of `simplex_lists` are done in `main` below.


class RosslerExperiment:
    def __init__(self, N, order, simplex_lists, M=150, tmax=20, device=None):
        self.N = N
        self.order = order
        self.simplex_lists = simplex_lists
        self.all_simplexes, self.true_simplex_sets = build_simplex_sets(N, order, simplex_lists)
        self.M = M
        self.tmax = tmax
        self.dt = tmax / M
        self.t_eval = np.linspace(0, tmax, M + 1)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scheduler = None

    def simulate(self, x0=None):
        if x0 is None:
            x0 = np.random.uniform(-1, 1, size=(3 * self.N,))
        sol = solve_ivp(roessler_hoi_extended, (0, self.tmax), x0, t_eval=self.t_eval, args=(self.simplex_lists, self.order))
        X = sol.y.T
        self.t_data = torch.tensor(self.t_eval, dtype=torch.float32).unsqueeze(1)
        self.x_data = torch.tensor(X, dtype=torch.float32)
        return sol

    def build_model(self, arch_name='ResNet', use_resnet=True, use_attention=False, lr=5e-4):
        arch_map = {'ResNet': (True, False), 'Attention': (False, True)}
        use_resnet, use_attention = arch_map.get(arch_name, (use_resnet, use_attention))
        self.model = HyperPINNTopology(N=self.N, output_dim=3 * self.N, use_resnet=use_resnet, use_attention=use_attention, max_order=self.order)
        self.model = self.model.to(self.device)
        for curr_order in range(2, self.order + 1):
            if curr_order == 2:
                self.model.lambda_l1[curr_order] = 0.03
                self.model.lambda_l0[curr_order] = 0.01
            elif curr_order == 3:
                self.model.lambda_l1[curr_order] = 0.05
                self.model.lambda_l0[curr_order] = 0.02
            else:
                factor = 1.5 ** (curr_order - 3)
                self.model.lambda_l1[curr_order] = 0.05 * factor
                self.model.lambda_l0[curr_order] = 0.02 * factor

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=14000, eta_min=1e-6)

    def prepare_tensors(self):
        self.t_data = self.t_data.float().to(self.device)
        self.x_data = self.x_data.float().to(self.device)

    def train(self, epochs=1000, stage1_epochs=250, stage2_epochs=750, adaptive_weights=True):
        if self.model is None:
            raise RuntimeError('Model not built. Call build_model() first.')

        losses = []
        sparsity_stats = []
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            x_pred = self.model.forward(self.t_data)
            physics_loss = self.model.physics_loss(self.t_data)
            data_loss = torch.mean((x_pred - self.x_data) ** 2)
            sparsity_loss, sparsity_info = self.model.sparsity_regularization(self.t_data)

            if adaptive_weights and epoch > 50:
                sparsity_weight = max(0.1, 1.0 * (0.99 ** (epoch - 50)))
            else:
                sparsity_weight = 1.0

            if epoch < stage1_epochs:
                physics_weight = 0.01
                data_weight = 1.0
                sparsity_weight = 0.0
            elif epoch < stage2_epochs:
                progress = (epoch - stage1_epochs) / max(1, (stage2_epochs - stage1_epochs))
                physics_weight = 0.01 + 0.99 * progress
                data_weight = 1.0 - 0.8 * progress
                sparsity_weight = 0.0
            else:
                progress = min(1.0, (epoch - stage2_epochs) / max(1, (epochs - stage2_epochs)))
                physics_weight = 1.0
                data_weight = 0.2
                sparsity_weight = 0.1 * progress

            total_loss = physics_weight * physics_loss + data_weight * data_loss + sparsity_weight * sparsity_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            losses.append(total_loss.item())
            sparsity_stats.append(sparsity_info)

            if epoch % max(1, epochs // 10) == 0:
                print(f'Epoch {epoch}, Loss: {total_loss.item():.6f}')

        return losses, sparsity_stats

    def evaluate(self):
        return self.evaluate_multi_order_interactions()

    def evaluate_multi_order_interactions(self):
        self.model.eval()
        with torch.no_grad():
            if self.t_data.device != next(self.model.parameters()).device:
                t_data = self.t_data.to(next(self.model.parameters()).device)
            else:
                t_data = self.t_data

            _, _, weights_by_order = self.model.get_sparse_weights(t_data, use_concrete=False, hard=False)

        y_true_all = []
        y_score_all = []
        results_by_order = {}

        for curr_order in range(2, self.order + 1):
            if curr_order not in self.all_simplexes or curr_order not in weights_by_order:
                continue

            y_true = []
            y_score = []
            all_curr_simplexes = self.all_simplexes[curr_order]
            true_set = self.true_simplex_sets.get(curr_order, set())
            probs = weights_by_order[curr_order][-1].cpu().numpy()

            for idx, simplex in enumerate(all_curr_simplexes):
                simplex_tuple = tuple(sorted(simplex))
                y_true.append(1 if simplex_tuple in true_set else 0)
                y_score.append(abs(probs[idx]) if idx < len(probs) else 0.0)

            results_by_order[curr_order] = (np.array(y_true), np.array(y_score))
            y_true_all.extend(y_true)
            y_score_all.extend(y_score)

        return np.array(y_true_all), np.array(y_score_all), results_by_order


if __name__ == '__main__':
    N = 8
    order = 6
    simplex_lists = default_simplex_lists(N, order)

    exp = RosslerExperiment(N=N, order=order, simplex_lists=simplex_lists, M=150, tmax=20)
    sol = exp.simulate()
    exp.build_model('ResNet')
    exp.prepare_tensors()
    print(f'Using device: {exp.device}')
    exp.train(epochs=10, stage1_epochs=2, stage2_epochs=6)
    y_true_all, y_score_all, results_by_order = exp.evaluate()
    print('Finished training/evaluation')
