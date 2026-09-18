import torch
import numpy as np
import config
from utils.metrics import evaluate_reconstruction, solve_hungarian_assignment

'''
4) synthetic data matching to biological data
    - This is difficult because 1) we dont know if an exact matching exists, 2) each network could be representing a different permutation
    - initially try the raw synthetic networks 
    - for any networks that do converge, isolate the hubs, and compare them to the empirical networks - probably none will converge properly though
    - calclate the number/percent of synthetic hubs that match to empirical hubs

'''

def generate_null_distribution(A_synth, A_empir_template, n_samples=1000):
    null_l1 = []
    n = A_synth.shape[0]
    for _ in range(n_samples):
        rand_p = torch.randperm(n)
        eval_res = evaluate_reconstruction(A_empir_template, A_synth, rand_p)
        null_l1.append(eval_res['l1'])
    return np.array(null_l1)

def evaluate_approx_alignment(model, A_synth, A_empir_template):
    with torch.no_grad():
        soft_matrix = model(A_synth.unsqueeze(0).to(config.DEVICE)).squeeze(0)
    hard_mapping = solve_hungarian_assignment(soft_matrix).cpu()

    match_eval = evaluate_reconstruction(A_empir_template, A_synth, hard_mapping)
    E_matched = match_eval['l1']

    null_l1_err = generate_null_distribution(A_synth, A_empir_template)
    E_random = np.mean(null_l1_err)

    Improve_score = (E_random - E_matched) / (E_random + 1e-8)
    p_val = (null_l1_err <= E_matched).mean()

    return hard_mapping, Improve_score, p_val


def calculate_hub_overlap(A_synth, A_empir, hard_mapping, top_k=10):
    synth_strengths = A_synth.sum(dim=-1)
    emp_strengths = A_empir.sum(dim=-1)

    top_synth = set(torch.argsort(synth_strengths, descending=True)[:top_k].tolist())
    top_emp = set(torch.argsort(emp_strengths, descending=True)[:top_k].tolist())

    mapped_synth_hubs = set(hard_mapping[list(top_synth)].tolist())
    overlap = len(mapped_synth_hubs.intersection(top_emp)) / float(top_k)
    return overlap

def run_phase_4():
    '''
    wrapper for the synthetic networks
    '''