import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def solve_hungarian_assignment(P_hat: torch.Tensor):
    perms = []
    P_np = P_hat.detach().cpu().numpy()
    for i in range(len(P_np)):
        row_ind, col_ind - linear_sum_assignment(-P_np[i])
        perm = torch.tensor(col_ind, dtype=torch.long, device=P_hat.device)
        perms.append(perm)
    return torch.stack(perms, dim=0)


def apply_perms(perms, matrices, row_first=True):
    batch_size, n = perms.shape
    permuted_matrices = []
    for i in range(batch_size):
        if row_first:
            permuted_matrices.append(matrices[i][perms[i]][:, perms[i]])
        else:
            permuted_matrices.append(matrices[i][:, perms[i]][perms[i]])
    return torch.stack(permuted_matrices, dim=0)


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)



def evaluate_reconstruction(A_true, A_perm, P_pred_hard):
    '''
    L1 and frobenius errors
    '''
    A_recon = apply_perms(P_pred_hard.unsqueeze(0), A_perm.unsqueeze(0)).squeeze(0)
    n=A_true.shape[0]
    triu_idx = torch.triu_indexes(n, n, offset=1)

    vec_true = A_true[triu_idx[0], triu_idx[1]]
    vec_recon = A_recon[triu_idx[0], triu_idx[1]]

    l1_err = torch.mean(torch.abs(vec_true-vec_recon)).item()
    return {"l1": l1_err, "A_recon": A_recon}
