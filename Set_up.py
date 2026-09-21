import pickle

import torch
import numpy as np
import config
from pathlib import Path

'''
1) Set up the data
    - Load the networks from SCmu.npy
    - standardise the edge weights to fixed range - this is useful for gradient stability across the architectures
    - Generate permutations from real data - this should be explicitly saved along with the seed integers
        - graph_id
        - permutation_seed
        - actual_permutation
        - inverse_permutation
    - Load also synthetic networks from the mecahnistic model 
'''

def load_connectomes(path=None):
    path = Path(path or config.REAL_DATA_PATH)
    graphs = torch.as_tensor(np.load(path, allow_pickle=True)).float()
    return validate_connectomes(graphs)

def load_synthetic_connectomes(path=None):
    path = Path(path or config.SYNTHETIC_DATA_PATH)
    if path.suffix == '.npy':
        graphs = torch.as_tensor(np.load(path, allow_pickle=True)).float()
    elif path.suffix == '.pkl':
        graphs = torch.load(path)
    with path.open('rb') as f:
        data = pickle.load(f)

    records  = list(data.values()) if isinstance(data, dict) else data
    matrices =[]
    for i, record in enumerate(records):
        if isinstance(record, torch.Tensor) or isinstance(record, np.ndarray):
            matrix = record
        elif isinstance(record, dict):
            matrix = record.get("adjacency_matrix", record.get("W"))
            if matrix is None and "output_dict" in record:
                values = record["output_dict"]
                candidates = values.values() if isinstance(values, dict) else values
                candidates = list(candidates)
                matrix = next(
                    (value for value in candidates
                     if np.asarray(value).shape ==
                     (config.NUM_NODES, config.NUM_NODES)),
                    None,
                )
        else:
            matrix = None

        if matrix is None:
            raise ValueError(f"Could not find adjacency matrix in record {i}")
        matrices.append(torch.as_tensor(matrix).float())
    return validate_connectomes(torch.stack(matrices))   


def validate_connectomes(tensor):
    if tensor.ndim != 3:
        raise ValueError("Expected [B, N, N] tensor")

    _, n, m = tensor.shape

    if n != config.NUM_NODES or m != config.NUM_NODES:
        raise ValueError(
            f"Expected [{config.NUM_NODES}, {config.NUM_NODES}] matrices"
        )

    if not torch.isfinite(tensor).all():
        raise ValueError("Matrix contains non-finite values")

    if (tensor < 0).any():
        raise ValueError("Negative edge weights found")

    if not torch.allclose(
        tensor, tensor.transpose(-1, -2), atol=1e-4
    ):
        raise ValueError("Matrices are not symmetric")

    if torch.diagonal(tensor, dim1=-2, dim2=-1).abs().max() > 1e-8:
        raise ValueError("Matrices must have zero diagonal")
    return tensor


def normalize_connectomes(graphs, train_idx=None):
    graphs = validate_connectomes(graphs)
    if train_idx is None:
        train_idx = torch.arange(len(graphs))
    train_idx = torch.as_tensor(train_idx, dtype=torch.long)
    train_max = graphs[train_idx].max().item()
    if train_max <= 0:
        raise ValueError("Training data contain no positive edge weights")
    norm_config = {
        "train_max": float(train_max),
        "method": "train_global_max",
    }
    return graphs / train_max, norm_config


def generate_permutations(graph_ids, seeds, split_name, output_path=None):
    records = []
    for graph_id in graph_ids:
        for seed in seeds:
            generator = torch.Generator().manual_seed(
                int(seed) + 1_000_003 * int(graph_id)
            )
            perm = torch.randperm(config.NUM_NODES, generator=generator)
            records.append({
                "graph_id": int(graph_id),
                "seed": int(seed),
                "perm": perm,
                "inverse_perm": torch.argsort(perm),
                "split": split_name,
            })

    output_path = Path(output_path or f"manifest_{split_name}.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, output_path)
    return records


def extract_structural_node_feats(adj_matrix, tol=1e-8):
    adj_matrix = torch.as_tensor(adj_matrix).float()
    strength = adj_matrix.sum(-1, keepdim=True)
    degree = (adj_matrix.abs() > tol).float().sum(-1, keepdim=True)
    mean_w = adj_matrix.mean(-1, keepdim=True)
    std_w = adj_matrix.std(-1, keepdim=True, unbiased=False)
    _, eigenvectors = torch.linalg.eigh(adj_matrix)
    centrality = eigenvectors[:, -1].abs().unsqueeze(-1)

    features = torch.cat(
        [strength, degree, mean_w, std_w, centrality], dim=-1
    )
    return (features - features.mean(0, keepdim=True)) / (
        features.std(0, keepdim=True, unbiased=False) + 1e-6
    )

def run_phase_1(real_path=None, synthetic_path=None, output_path=None, manifest_dir=".",):
    '''
    wrapper for saving preprocessed_data.pt
    '''
    real_graphs = load_real_graphs(real_path)
    real_indices = torch.arange(len(real_graphs))
    real_graphs, norm_config = normalize_connectomes(
        real_graphs, real_indices
    )

    synthetic_graphs = None
    if synthetic_path or config.SYNTHETIC_DATA_PATH not in {None, "", ".pkl"}:
        synthetic_graphs = load_synthetic_graphs(synthetic_path)
        synthetic_graphs = synthetic_graphs / norm_config["train_max"]

    for split_name, seeds in (
        ("seen", config.SEEDS_SEEN),
        ("finetune", config.SEEDS_FINETUNE),
        ("unseen", config.SEEDS_UNSEEN),
    ):
        generate_permutations(
            graph_ids=range(len(real_graphs)),
            seeds=seeds,
            split_name=split_name,
            output_path=Path(manifest_dir) / f"manifest_{split_name}.pt",
        )

    payload = {
        "real_graphs": real_graphs,
        "synthetic_graphs": synthetic_graphs,
        "normalization": norm_config,
        "real_features": torch.stack([
            extract_structural_node_feats(graph)
            for graph in real_graphs
        ]),
    }
    output_path = Path(output_path or config.PREPROCESSED_DATA_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return payload


if __name__ == "__main__":
    run_phase_1()   
