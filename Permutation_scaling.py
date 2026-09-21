
from pathlib import Path
import csv
import time
import numpy as np
import torch

import config
from models import structural_features, hard_assignment
from utils.metrics import evaluate_reconstruction



'''
3) Complexity and permutation scaling
    - inference - we wrap forward and hungarian to time.time() blobk and record execution time per graph
    - we try scaling to new permutation limits
    - we output the permutation pool size, the alighnemt accuracy, adn the mean inference time and we can see the bottleneck as hungarian has complexity O(N^3)
    - We do additional experiemtns to see if there is a faster way :
        - Baseline - fixed set of permutations (save the seeds) - eg (0-9) 
        - Fine-tuning - fine-tune on a new set of seeds - eg (10-19)
        - Unseen - Evaluate the model on unseen permutations - eg (20-29)
            - On the baseline model
            - on the fine-tuned model
    - we track the Hungarian accuracy and L1 across the three (3.5) scenarios

'''

def sync_dev(device=None):
    if config.DEVICE.type == 'mps':
        torch.mps.synchronize()
    elif config.DEVICE.type == 'cuda':
        torch.cuda.synchronize()


def predict_scores(model, A_true, A_perm, model_kind="mlp", device=None):
    device = device or config.DEVICE
    model = model.to(device).eval()
    A_true_d, A_perm_d = A_true.to(device), A_perm.to(device)

    with torch.no_grad():
        if model_kind == "mlp":
            scores = model(A_perm_d.unsqueeze(0))
        elif model_kind in {"gin", "gat", "gnn"}:
            f_true = structural_features(A_true_d)
            f_perm = structural_features(A_perm_d)
            scores = model(A_true_d, A_perm_d, f_true, f_perm)
        else:
            raise ValueError("model_kind must be 'mlp', 'gin', 'gat', or 'gnn'")
    return scores



def evaluate_with_timing(model, A_true, A_perm, expected=None, model_kind="mlp", device=None):
    A_perm = A_perm.to(config.DEVICE)
    timing = {}

    sync_dev()
    t0 = time.perf_counter()
    with torch.no_grad():
        soft_assignment = model(A_perm.unsqueeze(0)).squeeze(0)
    sync_dev()
    timing['forward_time'] = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred_mapping = solve_hungarian_assignment(soft_assignment.unsqueeze(0))[0]
    timing['hungarian time'] = time.perf_counter() - t1
    timing['total time'] = timing['forward_time'] + timing['hungarian time']

    return pred_mapping, timing

def load_manifest(manifest_path):
    records = torch.load(Path(manifest_path), weights_only=False)
    if not isinstance(records, (list, tuple)):
        raise ValueError("Manifest must contain a list of records")
    return records

def pair_from_record(graphs, record):
    graph_id = int(record["graph_id"])
    A_true = torch.as_tensor(graphs[graph_id]).float()
    perm = torch.as_tensor(record["perm"]).long()
    A_perm = A_true[perm][:, perm]
    expected = record.get("inverse_perm", record.get("inv_perm"))
    if expected is None:
        expected = torch.argsort(perm)
    return A_true, A_perm, torch.as_tensor(expected).long()

def evaluate_manifest(model, graphs, manifest_path, model_kind="mlp", device=None, label="test"):
    records = load_manifest(manifest_path)
    rows = []
    for record in records:
        A_true, A_perm, expected = pair_from_record(graphs, record)
        _, metrics = evaluate_with_timing(
            model, A_true, A_perm, expected,
            model_kind=model_kind, device=device,
        )
        rows.append({
            "split": label,
            "graph_id": int(record["graph_id"]),
            "seed": int(record["seed"]),
            "model_kind": model_kind,
            **metrics,
        })
    return rows

# def finetune_model(model, train_graphs, manifest_path, epochs=10, lr=1e-5, device=None, seed=0):
#     device = device or config.DEVICE
#     records = load_manifest(manifest_path)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     model = model.to(device).train()
#     generator = torch.Generator().manual_seed(int(seed))

#     for epoch in range(int(epochs)):
#         order = torch.randperm(len(records), generator=generator).tolist()
#         for idx in order:
#             A_true, A_perm, expected = pair_from_record(train_graphs, records[idx])
#             A_true, A_perm, expected = A_true.to(device), A_perm.to(device), expected.to(device)
#             soft_assignment = model(A_perm.unsqueeze(0))[0]
#             reconstruction = soft_assignment @ A_perm @ soft_assignment.T
#             loss = (A_true - reconstruction).abs().mean()
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#     model.eval()
#     return model

def finetune_model(model, train_graphs, manifest_path, model_kind="mlp",
                   epochs=10, lr=None, temperature=1.0, device=None, seed=0):

    if model_kind == "mlp":
        return finetune_mlp(
            model, train_graphs, manifest_path, epochs=epochs,
            lr=1e-5 if lr is None else lr, device=device, seed=seed,
        )
    if model_kind in {"gin", "gat", "gnn"}:
        return finetune_gnn(
            model, train_graphs, manifest_path, epochs=epochs,
            lr=1e-4 if lr is None else lr, temperature=temperature,
            device=device, seed=seed,
        )
    raise ValueError("model_kind must be 'mlp', 'gin', 'gat', or 'gnn'")


def finetune_mlp(model, train_graphs, manifest_path, epochs=10,
                 lr=1e-5, device=None, seed=0):
    device = device or config.DEVICE
    records = load_manifest(manifest_path)
    if not records:
        raise ValueError("Fine-tuning manifest is empty")

    model = model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(int(seed))

    for _ in range(int(epochs)):
        order = torch.randperm(len(records), generator=generator).tolist()
        for idx in order:
            A_true, A_perm, _ = pair_from_record(train_graphs, records[idx])
            A_true, A_perm = A_true.to(device), A_perm.to(device)
            soft_P = model(A_perm.unsqueeze(0))[0]
            reconstruction = soft_P @ A_perm @ soft_P.T
            loss = (A_true - reconstruction).abs().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval()


def finetune_gnn(model, train_graphs, manifest_path, epochs=10,
                 lr=1e-4, temperature=1.0, device=None, seed=0):
    device = device or config.DEVICE
    records = load_manifest(manifest_path)
    if not records:
        raise ValueError("Fine-tuning manifest is empty")

    model = model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(int(seed))

    for _ in range(int(epochs)):
        order = torch.randperm(len(records), generator=generator).tolist()
        for idx in order:
            A_true, A_perm, target = pair_from_record(train_graphs, records[idx])
            A_true, A_perm, target = (
                A_true.to(device), A_perm.to(device), target.to(device)
            )
            f_true = structural_features(A_true)
            f_perm = structural_features(A_perm)
            similarity = model(A_true, A_perm, f_true, f_perm)
            loss = correspondence_loss(similarity / temperature, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval()

def correspondence_loss(similarity, target):
    if similarity.ndim != 2:
        raise ValueError("similarity must have shape [N, N]")
    source_to_true = torch.argsort(target)
    row_loss = F.cross_entropy(similarity, target)
    col_loss = F.cross_entropy(similarity.T, source_to_true)
    return 0.5 * (row_loss + col_loss)


def run_generalisation_test(baseline_model, finetuned_model, graphs, seen_manifest, finetune_manifest,
 unseen_manifest, model_kind="mlp", device=None):
    rows = []
    cases = (
        ("baseline_seen", baseline_model, seen_manifest),
        ("baseline_finetune_set", baseline_model, finetune_manifest),
        ("baseline_unseen", baseline_model, unseen_manifest),
        ("finetuned_unseen", finetuned_model, unseen_manifest),
    )
    for label, model, manifest in cases:
        rows.extend(evaluate_manifest(
            model, graphs, manifest, model_kind=model_kind,
            device=device, label=label,
        ))
    return rows

def summarize_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["split"], row["model_kind"]), []).append(row)

    summary = []
    for (split, model_kind), values in grouped.items():
        def mean(key):
            xs = [v[key] for v in values if v.get(key) is not None]
            return float(np.mean(xs)) if xs else float("nan")
        summary.append({
            "split": split,
            "model_kind": model_kind,
            "n_pairs": len(values),
            "mean_accuracy": mean("accuracy"),
            "mean_l1": mean("l1"),
            "mean_frobenius": mean("frobenius"),
            "mean_forward_time": mean("forward_time"),
            "mean_hungarian_time": mean("hungarian_time"),
            "mean_total_time": mean("total_time"),
        })
    return summary



def save_rows(rows, output_path):
    if not rows:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_phase_3(model=None, finetuned_model=None, graphs=None, seen_manifest="manifest_seen.pt", finetune_manifest="manifest_finetune.pt", unseen_manifest="manifest_unseen.pt", model_kind="mlp", device=None, output_path="results/permutation_scaling.csv"):
    '''
    wrapper for outputting file post_matching_accuracies.csv
    '''
    if model is None or graphs is None:
            raise ValueError("Pass a baseline model and graphs")
    if finetuned_model is None:
        finetuned_model = model

    rows = run_generalisation_test(
        model, finetuned_model, graphs,
        seen_manifest, finetune_manifest, unseen_manifest,
        model_kind=model_kind, device=device,
    )
    summary = summarize_rows(rows)
    save_rows(rows, output_path)
    save_rows(summary, Path(output_path).with_name(
        Path(output_path).stem + "_summary.csv"))
    return rows, summary

if __name__ == "__main__":
    run_phase_3()   
