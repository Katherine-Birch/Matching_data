from pathlib import Path
import csv
import torch
from scipy.optimize import quadratic_assignment

from models import structural_features, hard_assignment
from utils.metrics import evaluate_reconstruction


'''
2)  Benchmarking
    - use the FAQ model using scipy.optimize.quadratic_assignment(method='faq')
    - global topology using MLP-Sinkhorn model - this minimises Frobenius and/or L1 norm between the predicted soft-permitation recon and the original - differentiable outputs soft permutation matrix wehre rows and columns sum to 1
    - local embeddings using GNN and GAT - this uses the cross-entropy loss of the node similarity matrix - differentiable where the output is continuous node similarity matrix
    - Hungarian algorityhm applied identically across MLP and GNN to extract final hard assignments - this is the non-differentiable part where the output is binary so we know where each node goes
'''

def run_faq_baseline(A_true, A_perm):
    """Return mapping true-node order -> permuted-node order."""
    result = quadratic_assignment(
        A_true.detach().cpu().numpy(),
        A_perm.detach().cpu().numpy(),
        method="faq",
        options={"maximize": False},
    )
    return torch.as_tensor(result.col_ind, dtype=torch.long)



def run_heuristic_baseline(A_true, A_perm, heuristic='centrality'):
    '''
    eigenvector centrality or strnght/degree
    '''
    if heuristic == 'strength':
        score_true = A_true.sum(dim=-1)
        score_perm = A_perm.sum(dim=-1)
    elif heuristic == 'degree':
        score_true = (A_true > 1e-8).float().sum(dim=-1)
        score_perm = (A_perm > 1e-8).float().sum(dim=-1)
    elif heuristic == 'centrality':
        _, e_true = torch.linalg.eigh(A_true)
        _, e_perm = torch.linalg.eigh(A_perm)
        score_true = e_true[:, -1].abs()
        score_perm = e_perm[:, -1].abs()
    else:
        raise ValueError(f"unknown heuristic: {heuristic}")

    order_true = torch.argsort(score_true, descending=True)
    order_perm = torch.argsort(score_perm, descending=True)

    mapping = torch.empty_like(order_true)
    mapping[order_true] = order_perm

    return mapping


def _hard_mapping_from_similarity(similarity):
    """Convert [N,N] or [1,N,N] scores to [N] assignment indices."""
    return hard_assignment(similarity)[0]

def evaluate_mapping(A_true, A_perm, mapping, expected=None):
    result = evaluate_reconstruction(A_true, A_perm, mapping)
    output = {
        "l1": result["l1"],
        "frobenius": result.get("frobenius"),
    }
    if expected is not None:
        output["accuracy"] = (
            mapping.detach().cpu() == expected.detach().cpu()
        ).float().mean().item()
    return output


def evaluate_pair(models_dict, A_true, A_perm, expected=None, device=None):
    """Evaluate all available methods on one known-permutation graph pair."""
    device = device or next(
        (p.device for m in models_dict.values() for p in m.parameters()),
        torch.device("cpu"),
    )
    A_true = A_true.float()
    A_perm = A_perm.float()
    results = {}

    methods = {
        "faq": run_faq_baseline(A_true, A_perm),
        "heuristic_strength": run_heuristic_baseline(
            A_true, A_perm, "strength"
        ),
        "heuristic_degree": run_heuristic_baseline(
            A_true, A_perm, "degree"
        ),
        "heuristic_centrality": run_heuristic_baseline(
            A_true, A_perm, "centrality"
        ),
    }

    for name, mapping in methods.items():
        results[name] = evaluate_mapping(
            A_true, A_perm, mapping, expected
        )

    if "mlp" in models_dict:
        model = models_dict["mlp"].to(device).eval()
        with torch.no_grad():
            soft_P = model(A_perm.to(device).unsqueeze(0))
        mapping = _hard_mapping_from_similarity(soft_P).cpu()
        results["mlp_sinkhorn"] = evaluate_mapping(
            A_true, A_perm, mapping, expected
        )

    for name in ("gin", "gat", "gnn"):
        if name not in models_dict:
            continue
        model = models_dict[name].to(device).eval()
        f_true = structural_features(A_true.to(device))
        f_perm = structural_features(A_perm.to(device))
        with torch.no_grad():
            similarity = model(
                A_true.to(device), A_perm.to(device), f_true, f_perm
            )
        mapping = _hard_mapping_from_similarity(similarity).cpu()
        results[name] = evaluate_mapping(
            A_true, A_perm, mapping, expected
        )

    return results

def eval_known_permutations(models_dict, graphs, manifest_path, device=None):
    """Shared evaluator for empirical or synthetic known-permutation tests."""
    records = torch.load(Path(manifest_path), weights_only=False)
    rows = []

    for record in records:
        graph_id = int(record.get("graph_id", 0))
        A_true = graphs[graph_id].float()
        perm = record["perm"].long()
        inverse_perm = record.get("inverse_perm", record.get("inv_perm"))
        if inverse_perm is None:
            inverse_perm = torch.argsort(perm)

        A_perm = A_true[perm][:, perm]
        scores = evaluate_pair(
            models_dict, A_true, A_perm, inverse_perm, device
        )

        for method, values in scores.items():
            rows.append({
                "graph_id": graph_id,
                "seed": int(record["seed"]),
                "method": method,
                **values,
            })

    return rows




def eval_task_a(models_dict, empirical_graphs, manifest_path, device=None):
    """Known-permutation benchmark on empirical graphs."""

    return eval_known_permutations(
        models_dict, empirical_graphs, manifest_path, device
    )

def eval_task_b(models_dict, synthetic_graphs, manifest_path, device=None):
    """Known-permutation benchmark on synthetic graphs.

    This tests recovery of a known latent synthetic ordering after applying
    an externally generated permutation. It does not test synthetic-to-
    empirical anatomical alignment; that is a separate analysis.
    """
    return eval_known_permutations(
        models_dict, synthetic_graphs, manifest_path, device
    )


def save_results(rows, output_path):
    if not rows:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)



    

def run_phase_2(
    models_dict=None,
    empirical_graphs=None,
    synthetic_graphs=None,
    empirical_manifest="manifest_seen.pt",
    synthetic_manifest=None,
    output_path="results/benchmark_results.csv",
    device=None,
):
    """Run empirical and, optionally, synthetic known-permutation tests."""
    models_dict = {} if models_dict is None else models_dict
    if empirical_graphs is None:
        raise ValueError("Pass empirical_graphs with shape [B, N, N]")

    rows = eval_task_a(
        models_dict, empirical_graphs, empirical_manifest, device
    )

    if synthetic_graphs is not None and synthetic_manifest is not None:
        synthetic_rows = eval_task_b(
            models_dict, synthetic_graphs, synthetic_manifest, device
        )
        for row in synthetic_rows:
            row["dataset"] = "synthetic"
        rows.extend(synthetic_rows)

    for row in rows:
        row.setdefault("dataset", "empirical")

    save_results(rows, output_path)
    return rows

if __name__ == "__main__":
    run_phase_2()   
