import time
import torch
import config
from utils.metrics import solve_hungarian_assignment, evaluate_reconstruction

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

def sync_dev():
    if config.DEVICE.type == 'mps':
        torch.mps.synchronize()
    elif config.DEVICE.type == 'cuda':
        torch.cuda.synchronize()

def evaluate_with_timing(model, A_true, A_perm, device):
    A_perm = A_perm.to(config.DEVICE)
    timing = {}

    sync_dev()
    t0 = time.perf_counter()
    with torch.no_grad():
        soft_assignment = model(A_perm.unsqueeze(0)).squeeze
    sync_dev()
    timing['forward_time'] = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred_mapping = solve_hungarian_assignment(soft_assignment)
    timing['hungarian time'] = time.perf_counter() - t1
    timing['total time'] = timing['forward_time'] + timing['hungarian time']

    return pred_mapping, timing

def finetune_model(model, train_graphs, finetune_manifest):
    


def run_generalisation_test(baseline_model, finetuned_model, test_graphs):

def run_phase_3():
    '''
    wrapper for outputting file post_matching_accuracies.csv
    '''

