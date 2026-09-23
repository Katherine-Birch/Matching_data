from pathlib import Path
import csv
from Set_up import load_connectomes
import numpy as np
import torch
from scipy.optimize import quadratic_assignment

import config
from models import structural_features, hard_assignment
from metrics import evaluate_reconstruction

'''
4) synthetic data matching to biological data
    - This is difficult because 1) we dont know if an exact matching exists, 2) each network could be representing a different permutation
    - initially try the raw synthetic networks 
    - for any networks that do converge, isolate the hubs, and compare them to the empirical networks - probably none will converge properly though
    - calclate the number/percent of synthetic hubs that match to empirical hubs

'''


def validate_pair(A_synth, A_empir, tol=1e-5):
    A_synth = torch.as_tensor(A_synth).float()
    A_empir = torch.as_tensor(A_empir).float()
    if A_synth.ndim != 2 or A_empir.ndim != 2:
        raise ValueError("Both networks must have shape [N, N]")
    if A_synth.shape != A_empir.shape:
        raise ValueError("Synthetic and empirical networks must have the same shape")
    if A_synth.shape != (config.NUM_NODES, config.NUM_NODES):
        raise ValueError(f"Expected [{config.NUM_NODES}, {config.NUM_NODES}]")
    if not torch.isfinite(A_synth).all() or not torch.isfinite(A_empir).all():
        raise ValueError("Networks contain non-finite values")
    if (A_synth < 0).any() or (A_empir < 0).any():
        raise ValueError("Networks contain negative weights")
    if not torch.allclose(A_synth, A_synth.T, atol=tol):
        raise ValueError("Synthetic network must be symmetric")
    if not torch.allclose(A_empir, A_empir.T, atol=tol):
        raise ValueError("Empirical network must be symmetric")
    return A_synth, A_empir


def generate_null_distribution(A_synth, A_empir_template, n_samples=1000, seed=0):
    A_synth, A_empir_template = validate_pair(A_synth, A_empir_template)
    generator = torch.Generator().manual_seed(int(seed))
    errors = []
    for _ in range(int(n_samples)):
        mapping = torch.randperm(A_synth.shape[0], generator=generator)
        errors.append(evaluate_reconstruction(
            A_empir_template, A_synth, mapping
        )["l1"])
    return np.asarray(errors, dtype=float)


def predict_mlp_mapping(model, A_synth, device):
    """Return target empirical index -> source synthetic index."""
    model = model.to(device).eval()
    with torch.no_grad():
        scores = model(A_synth.to(device).unsqueeze(0))
    return hard_assignment(scores)[0].cpu()


def predict_gnn_mapping(model, A_synth, A_empir, device):
    """Return target empirical index -> source synthetic index."""
    model = model.to(device).eval()
    f_synth = structural_features(A_synth.to(device))
    f_empir = structural_features(A_empir.to(device))
    with torch.no_grad():
        similarity = model(
            A_empir.to(device), A_synth.to(device), f_empir, f_synth
        )
    return hard_assignment(similarity)[0].cpu()


def run_faq_alignment(A_synth, A_empir_template):
    """FAQ mapping in the target empirical -> source synthetic convention."""
    A_synth, A_empir_template = validate_pair(A_synth, A_empir_template)
    result = quadratic_assignment(
        A_empir_template.numpy(),
        A_synth.numpy(),
        method="faq",
        options={"maximize": False},
    )
    return torch.as_tensor(result.col_ind, dtype=torch.long)


def predict_mapping(model, A_synth, A_empir_template, model_kind="mlp", device=None):
    A_synth, A_empir_template = validate_pair(A_synth, A_empir_template)
    device = device or config.DEVICE
    if model_kind == "faq":
        return run_faq_alignment(A_synth, A_empir_template)
    if model_kind == "mlp":
        if model is None:
            raise ValueError("Pass a trained MLP model")
        return predict_mlp_mapping(model, A_synth, device)
    if model_kind in {"gin", "gat", "gnn"}:
        if model is None:
            raise ValueError("Pass a trained GNN/GAT model")
        return predict_gnn_mapping(model, A_synth, A_empir_template, device)
    raise ValueError("model_kind must be 'mlp', 'gin', 'gat', 'gnn', or 'faq'")


def calculate_hub_overlap(A_synth, A_empir, target_to_source, top_k=10):
    """Overlap of top-strength hubs after target->source mapping.

    target_to_source[target] = source, so the mapping is inverted before
    synthetic source nodes are mapped into empirical target-node space.
    """
    A_synth, A_empir = validate_pair(A_synth, A_empir)
    target_to_source = torch.as_tensor(target_to_source).long().cpu()
    n = A_synth.shape[0]
    top_k = min(int(top_k), n)

    synth_hubs = set(torch.argsort(
        A_synth.sum(-1), descending=True
    )[:top_k].tolist())
    empir_hubs = set(torch.argsort(
        A_empir.sum(-1), descending=True
    )[:top_k].tolist())

    source_to_target = torch.argsort(target_to_source)
    mapped_synth_hubs = {
        source_to_target[source].item() for source in synth_hubs
    }
    return len(mapped_synth_hubs & empir_hubs) / float(top_k)


def hub_overlap_null(A_synth, A_empir, n_samples=1000, top_k=10, seed=0):
    A_synth, A_empir = validate_pair(A_synth, A_empir)
    generator = torch.Generator().manual_seed(int(seed))
    values = []
    for _ in range(int(n_samples)):
        mapping = torch.randperm(A_synth.shape[0], generator=generator)
        values.append(calculate_hub_overlap(
            A_synth, A_empir, mapping, top_k
        ))
    return np.asarray(values, dtype=float)


def evaluate_approx_alignment(
    model,
    A_synth,
    A_empir_template,
    model_kind="mlp",
    device=None,
    n_nulls=1000,
    seed=0,
    top_k=10,
):
    """Evaluate approximate alignment; no exact anatomical accuracy exists."""
    A_synth, A_empir_template = validate_pair(A_synth, A_empir_template)
    mapping = predict_mapping(
        model, A_synth, A_empir_template,
        model_kind=model_kind, device=device,
    )

    matched = evaluate_reconstruction(
        A_empir_template, A_synth, mapping
    )
    null_errors = generate_null_distribution(
        A_synth, A_empir_template, n_samples=n_nulls, seed=seed
    )
    random_error = float(null_errors.mean())
    improvement = (random_error - matched["l1"]) / (random_error + 1e-12)
    p_value = (
        1 + np.sum(null_errors <= matched["l1"])
    ) / (len(null_errors) + 1)

    overlap = calculate_hub_overlap(
        A_synth, A_empir_template, mapping, top_k=top_k
    )
    overlap_null = hub_overlap_null(
        A_synth, A_empir_template,
        n_samples=n_nulls, top_k=top_k, seed=seed + 1,
    )
    overlap_p = (
        1 + np.sum(overlap_null >= overlap)
    ) / (len(overlap_null) + 1)

    return {
        "mapping": mapping,
        "l1": matched["l1"],
        "frobenius": matched.get("frobenius"),
        "random_l1_mean": random_error,
        "alignment_improvement": float(improvement),
        "alignment_p_value": float(p_value),
        "hub_overlap": float(overlap),
        "hub_overlap_null_mean": float(overlap_null.mean()),
        "hub_overlap_p_value": float(overlap_p),
    }


def mean_empirical_template(empirical_graphs):
    empirical_graphs = torch.as_tensor(empirical_graphs).float()
    if empirical_graphs.ndim != 3:
        raise ValueError("empirical_graphs must have shape [B, N, N]")
    empirical_graphs = empirical_graphs.clone()
    template = empirical_graphs.mean(dim=0)
    template.fill_diagonal_(0)
    return template


def save_alignment_results(rows, output_path):
    if not rows:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_alignment_mappings(records, mapping_path):
    mapping_path = Path(mapping_path)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, mapping_path)



def run_phase_4(
    model=None,
    synthetic_graphs=None,
    empirical_graphs=None,
    model_kind="mlp",
    empirical_template=None,
    output_path="results/synthetic_alignment.csv",
    mapping_path=None,
    device=None,
    n_nulls=1000,
    top_k=10,
):
    """Run topology-based synthetic-to-empirical alignment."""
    if model_kind != "faq" and model is None:
        raise ValueError("Pass a trained model unless model_kind='faq'")
    if synthetic_graphs is None:
        raise ValueError("Pass synthetic_graphs with shape [B, N, N]")

    synthetic_graphs = torch.as_tensor(synthetic_graphs).float()
    if synthetic_graphs.ndim != 3:
        raise ValueError("synthetic_graphs must have shape [B, N, N]")

    if empirical_template is None:
        if empirical_graphs is None:
            raise ValueError("Pass empirical_graphs or empirical_template")
        empirical_template = mean_empirical_template(empirical_graphs)
    else:
        empirical_template = torch.as_tensor(empirical_template).float()

    output_path = Path(output_path)
    if mapping_path is None:
        mapping_path = output_path.with_name(
            output_path.stem + "_mappings.pt"
        )

    rows = []
    mapping_records = []
    for graph_id, A_synth in enumerate(synthetic_graphs):
        result = evaluate_approx_alignment(
            model=model,
            A_synth=A_synth,
            A_empir_template=empirical_template,
            model_kind=model_kind,
            device=device,
            n_nulls=n_nulls,
            seed=graph_id,
            top_k=top_k,
        )
        rows.append({
            "graph_id": graph_id,
            "model": model_kind,
            "l1": result["l1"],
            "frobenius": result["frobenius"],
            "random_l1_mean": result["random_l1_mean"],
            "alignment_improvement": result["alignment_improvement"],
            "alignment_p_value": result["alignment_p_value"],
            "hub_overlap": result["hub_overlap"],
            "hub_overlap_null_mean": result["hub_overlap_null_mean"],
            "hub_overlap_p_value": result["hub_overlap_p_value"],
        })
        mapping_records.append({
            "graph_id": graph_id,
            "model": model_kind,
            "mapping": result["mapping"].cpu(),
        })

    save_alignment_results(rows, output_path)
    save_alignment_mappings(mapping_records, mapping_path)
    print(f"Saved alignment metrics to {output_path}")
    print(f"Saved alignment mappings to {mapping_path}")
    return rows


if __name__ == "__main__":
    run_phase_4()