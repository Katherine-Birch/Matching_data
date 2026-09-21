import torch
from torch_geometric.nn import GINEConv, GATv2Conv
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment



def sample_gumbel(shape, device=None, eps=1e-20):
    u = torch.rand(shape, device=device).clamp_min(eps).clamp_max(1.0 - eps)
    return -torch.log(-torch.log(u))



def sinkhorn(logits, num_iters=20):
    for _ in range(num_iters):
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)
    return logits.exp()
                        
class MLPGumbelSinkhorn(torch.nn.Module):
    def __init__(self, num_nodes=90, hidden_dim1=512, hidden_dim2=256, tau=0.1, n_iters=20, gumbel_noise=True):
        super().__init__()
        self.num_nodes = num_nodes
        self.tau = tau
        self.n_iters = n_iters
        self.gumbel_noise = gumbel_noise
        graph_dim = num_nodes * num_nodes
        self.net = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, graph_dim)
        )
        

        def forward(self, adj, hard=False):
            logits = self.net(adj.flatten(1)).view(
                -1, self.num_nodes, self.num_nodes
            )
            if self.training and self.gumbel_noise:
                logits = logits + sample_gumbel(logits.shape, logits.device)
            soft = sinkhorn(logits / self.tau, self.n_iters)
            if not hard:
                return soft
            hard_perm = hard_assignment(soft)
            hard_matrix = F.one_hot(
                hard_perm, num_classes=self.num_nodes
            ).to(soft.dtype)
            # Straight-through estimator
            return hard_matrix + soft - soft.detach()



class GINEncoder(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=64, out_channels=64):
        super().__init__()
        mlp1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        mlp2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )
        self.conv1 = GINEConv(mlp1, edge_dim=1)
        self.conv2 = GINEConv(mlp2, edge_dim=1)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        return self.conv2(x, edge_index, edge_attr)



def dense_to_pyg(adj_matrix, node_features):
    if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("adj must have shape [N, N]")
    if node_features.ndim != 2 or node_features.shape[0] != adj_matrix.shape[0]:
        raise ValueError("node_features must have shape [N, F]")
    if not torch.allclose(
        adj_matrix, adj_matrix.T, atol=tol
    ):
        raise ValueError("adj_matrix must be symmetric")


    edge_index = torch.nonzero(adj_matrix != 0, as_tuple=False).t().contiguous()
    edge_attr = adj_matrix[edge_index[0], edge_index[1]].float().unsqueeze(-1)
    return node_features.float(), edge_index, edge_attr


class GATEncoder(nn.Module):
    """GATv2 encoder using weighted edges through edge_dim=1."""

    def __init__(self, in_channels=5, hidden_channels=64, out_channels=64,
                 heads=4, dropout=0.2):
        super().__init__()
        self.conv1 = GATv2Conv(
            in_channels, hidden_channels, heads=heads,
            edge_dim=1, dropout=dropout
        )
        self.conv2 = GATv2Conv(
            hidden_channels * heads, out_channels, heads=heads,
            concat=False, edge_dim=1, dropout=dropout
        )

    def forward(self, x, edge_index, edge_attr):
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        return self.conv2(x, edge_index, edge_attr)

class GraphMatchingModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def encode(self, adj, features):
        x, edge_index, edge_attr = dense_to_pyg(adj, features)
        z = self.encoder(x, edge_index, edge_attr)
        return F.normalize(z, dim=-1)

    def forward(self, adj_true, adj_perm, features_true, features_perm):
        z_true = self.encode(adj_true, features_true)
        z_perm = self.encode(adj_perm, features_perm)
        return z_true @ z_perm.T



def structural_features(adj):

    strength = adj.sum(dim=-1, keepdim=True)
    degree = (adj != 0).float().sum(dim=-1, keepdim=True)
    mean_w = adj.mean(dim=-1, keepdim=True)
    std_w = adj.std(dim=-1, keepdim=True)
 
    _, eigvecs = torch.linalg.eigh(adj)
    centrality = eigvecs[:, -1].abs().unsqueeze(-1)
    feats = torch.cat([strength, degree, mean_w, std_w, centrality], dim=-1)
    return (feats - feats.mean(0, keepdim=True)) / (feats.std(0, keepdim=True) + 1e-6)


def hard_assignment(similarity):
    if similarity.ndim == 2:
        similarity = similarity.unsqueeze(0)
    out = []
    for s in similarity.detach().cpu():
        _, cols = linear_sum_assignment(-s.numpy())
        out.append(torch.as_tensor(cols, dtype=torch.long, device=similarity.device))
    return torch.stack(out)


def structural_heuristic_assignment(adj_true, adj_perm, mode="strength"):
    f_true = structural_features(adj_true)
    f_perm = structural_features(adj_perm)
    if mode == "strength":
        f_true, f_perm = f_true[:, :1], f_perm[:, :1]
    elif mode == "degree":
        f_true, f_perm = f_true[:, 1:2], f_perm[:, 1:2]
    elif mode == "centrality":
        f_true, f_perm = f_true[:, 4:5], f_perm[:, 4:5]
    elif mode != "all":
        raise ValueError("mode must be strength, degree, centrality, or all")
    cost = torch.cdist(f_true, f_perm).detach().cpu().numpy()
    _, cols = linear_sum_assignment(cost)
    return torch.as_tensor(cols, dtype=torch.long, device=adj_true.device)