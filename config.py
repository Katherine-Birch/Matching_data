import torch

# Global Settings
NUM_NODES = 90
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
RANDOM_SEED = 42

# File Paths
REAL_DATA_PATH = 'SCmu.npy'
SYNTHETIC_DATA_PATH = '.pkl'
PREPROCESSED_DATA_PATH = '.pt'

# Permutation Seeds
SEEDS_SEEN = list(range(0, 10))
SEEDS_FINETUNE = list(range(10, 20))
SEEDS_UNSEEN = list(range(20, 30))