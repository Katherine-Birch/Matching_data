import torch
import numpy as np
import config

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

def validate_connectomes(tensor):
    b, n, m = tensor.shape
    assert n == m == config.NUM_NODES, f"expected {config.NUM_NODES}x{config.NUM_NODES}"
    assert torch.all(torch.isfinite(tensor)), "matrix contains non finite values"
    assert (tensor >= 0).all(), "negative values for edge weights"
    asym= (tensor - tensor.transpose(-1, -2)).abs().max()
    assert asym < 1e-4, f"matrixes are non sympetric max diff = {asym}"


def normalize_connectomes(graphs, train_idx):
    train_max = graphs[train_idx].max().item()
    norm_config = {"train_max": train_max, "method": "train_global_max"}
    return graphs / train_max, norm_config

def generate_permutations(seeds, split_name):
    records = []
    for s in seeds:
        g = torch.Generator().manual_seed(int(s))
        perm = torch.randperm(config.NUM_NODES, generator=g)
        records.append({"seed": int(s), "perm": perm, "inv_perm": torch.argsort(perm)})
    torch.save(records, f"manifest_{split_name}.pt")

def extract_structural_node_feats(adj_matrix):
    strength = adj_matrix.sum(dim=-1, keepdim=True)
    degrees = (adj_matrix>0).float().sum(dim=-1, keepdim=True)
    evals, evecs = torch.linalg.eigh(adj_matrix)
    centrality = evecs[:, -1:].abs()
    mean_w = adj_matrix.mean(dim=-1, keepdim=True)
    std_w = adj_matrix.std(dim=-1, keepdim=True)

    features = torch.cat([strength, degrees, centrality, mean_w, std_w], dim=-1)
    features = (features - features.mean(dim=0))/(features.std(dim=0) + 1e-6)
    return features

def run_phase_1():
    '''
    wrapper for saving preprocessed_data.pt
    '''

