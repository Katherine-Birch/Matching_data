# import torch
# import numpy as np
# from scipy.optimize import linear_sum_assignment

# def solve_hungarian_assignment(P_hat):
#     if P_hat.ndim == 2:
#         P_hat = P_hat.unsqueeze(0)

#     perms = []
#     for matrix in P_hat.detach().cpu().numpy():
#         _, cols = linear_sum_assignment(-matrix)
#         perms.append(torch.as_tensor(cols, dtype=torch.long))

#     return torch.stack(perms).to(P_hat.device)

# def apply_perms(perms, matrices, row_first=True):
#     batch_size, n = perms.shape
#     permuted_matrices = []
#     for i in range(batch_size):
#         if row_first:
#             permuted_matrices.append(matrices[i][perms[i]][:, perms[i]])
#         else:
#             permuted_matrices.append(matrices[i][:, perms[i]][perms[i]])
#     return torch.stack(permuted_matrices, dim=0)



# def evaluate_reconstruction(A_true, A_perm, P_pred_hard):
#     if P_pred_hard.ndim != 1:
#         raise ValueError("P_pred_hard must have shape [N]")

#     A_recon = A_perm[P_pred_hard][:, P_pred_hard]
#     idx = torch.triu_indices(
#         A_true.shape[0],
#         A_true.shape[0],
#         offset=1
#     )

#     x = A_true[idx[0], idx[1]]
#     y = A_recon[idx[0], idx[1]]
#     diff = x - y

#     return {
#         "l1": diff.abs().mean().item(),
#         "frobenius": torch.linalg.vector_norm(diff).item(),
#         "correlation": torch.corrcoef(
#             torch.stack((x, y))
#         )[0, 1].item(),
#         "A_recon": A_recon,
#     }

import torch
from scipy.optimize import linear_sum_assignment


def solve_hungarian_assignment(P_hat):
    if P_hat.ndim == 2:
        P_hat = P_hat.unsqueeze(0)

    assignments = []
    for scores in P_hat.detach().cpu().numpy():
        _, cols = linear_sum_assignment(-scores)
        assignments.append(torch.as_tensor(cols, dtype=torch.long))

    return torch.stack(assignments)


def apply_perms(perms, matrices):
    if perms.ndim == 1:
        perms = perms.unsqueeze(0)
    if matrices.ndim == 2:
        matrices = matrices.unsqueeze(0)

    matrices = matrices.cpu()
    perms = perms.cpu()
    return torch.stack([
        A[p][:, p] for A, p in zip(matrices, perms)
    ])


def evaluate_reconstruction(A_true, A_perm, P_pred_hard):
    A_true = torch.as_tensor(A_true).float().cpu()
    A_perm = torch.as_tensor(A_perm).float().cpu()
    P_pred_hard = torch.as_tensor(P_pred_hard).long().cpu()

    if P_pred_hard.ndim == 2:
        P_pred_hard = P_pred_hard[0]

    A_recon = A_perm[P_pred_hard][:, P_pred_hard]

    idx = torch.triu_indices(
        A_true.shape[0],
        A_true.shape[0],
        offset=1,
    )
    true_edges = A_true[idx[0], idx[1]]
    recon_edges = A_recon[idx[0], idx[1]]
    diff = true_edges - recon_edges

    result = {
        "l1": diff.abs().mean().item(),
        "frobenius": torch.linalg.vector_norm(diff).item(),
        "A_recon": A_recon,
    }

    if true_edges.numel() > 1 and true_edges.std() > 0 and recon_edges.std() > 0:
        result["correlation"] = torch.corrcoef(
            torch.stack([true_edges, recon_edges])
        )[0, 1].item()
    else:
        result["correlation"] = float("nan")

    return result