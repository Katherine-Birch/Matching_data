import torch
from scipy.optimize import quadratic_assignment

'''
2)  Benchmarking
    - use the FAQ model using scipy.optimize.quadratic_assignment(method='faq')
    - global topology using MLP-Sinkhorn model - this minimises Frobenius and/or L1 norm between the predicted soft-permitation recon and the original - differentiable outputs soft permutation matrix wehre rows and columns sum to 1
    - local embeddings using GNN and GAT - this uses the cross-entropy loss of the node similarity matrix - differentiable where the output is continuous node similarity matrix
    - Hungarian algorityhm applied identically across MLP and GNN to extract final hard assignments - this is the non-differentiable part where the output is binary so we know where each node goes
'''

def run_faq_baseline(A_true, A_perm):
    res = quadratic_assignment(A_true.cpu().numpy(), A_perm.cpu().numpy(), method='faq', options={'maximize': False})
    perm_recovered = torch.tensor(res.col_ind, dtype=torch.long)
    return perm_recovered


def run_heuristic_baseline(A_true, A_perm, heuristic='centrality'):
    '''
    eigenvector centrality or strnght/degree
    '''
    if heuristic == 'strength':
        score_true = A_true.sum(dim=-1)
        score_perm = A_perm.sum(dim=-1)
    elif heuristic == 'degree':
        score_true = (A_true > 0).float().sum(dim=-1)
        score_perm = (A_perm > 0).float().sum(dim=-1)
    elif heuristic == 'centrality':
        _, e_true = torch.linalg.eigh(A_true)
        _, e_perm = torch.linalg.eigh(A_perm)
        score_true = e_true[:, -1].abs()
        score_perm = e_perm[:, -1].abs()
    else:
        raise ValueError(f"unknown heuristic: {heuristic}")

    order_true = torch.argsort(score_true, descending=True)
    order_perm = torch.argsort(score_perm, descending=True)

    mapping = torch.zeros(A_true.shape[0], dtype=torch.long, device=A_true.device)
    mapping[order_true] = order_perm

    return mapping


def eval_task_a(models_dict, empirical_graphs, manifest):
    manifest = torch.load(manifest.path)

def eval_task_b(models_dict, synthetic_graphs, manifest):
    

def run_phase_2():
    '''
    wrapper for saving baseline accuracy tables
    '''