from pathlib import Path
import numpy as np

term_file = '/Users/kb01352/Downloads/Template_for_submission_to_Network_Neuroscience__NETN_/Code_files/final_figure/synthetic_term_matrices_FINAL.npy'
    
preterm_file = '/Users/kb01352/Downloads/Template_for_submission_to_Network_Neuroscience__NETN_/Code_files/final_figure/synthetic_preterm_matrices_FINAL.npy'

output_dir = Path("/Users/kb01352/Matching_data")   
combined_path = output_dir / "synthetic_matrices_combined.npy"
labels_path = output_dir / "synthetic_matrices_labels.npy"

term = np.load(term_path)
preterm = np.load(preterm_path)

if term.ndim != 3 or preterm.ndim != 3:
    raise ValueError(
        f"Expected arrays with shape [B, N, N]; got {term.shape} and {preterm.shape}"
    )
if term.shape[1:] != preterm.shape[1:]:
    raise ValueError(
        f"Network shapes do not match: {term.shape[1:]} vs {preterm.shape[1:]}"
    )

combined = np.concatenate([preterm, term], axis=0)
labels = np.concatenate([
    np.full(len(preterm), "preterm"),
    np.full(len(term), "term"),
])

np.save(combined_path, combined)
np.save(labels_path, labels)

print(f"Preterm: {preterm.shape}")
print(f"Term:    {term.shape}")
print(f"Combined: {combined.shape}")
print(f"Saved matrices: {combined_path}")
print(f"Saved labels:   {labels_path}")


# for real
# from pathlib import Path
# import numpy as np
# import pickle

# # CORRECTED: File variables now point to the correct files
# term_file = 'SCmu_term_real.pkl'    
# preterm_file = 'SCmu_preterm_real.pkl'

# output_dir = Path("/Users/kb01352/Matching_data")   
# # ADDED: Ensure the output directory actually exists before trying to save to it
# output_dir.mkdir(parents=True, exist_ok=True)

# combined_path = output_dir / "real_matrices_combined.npy"
# labels_path = output_dir / "real_matrices_labels.npy"

# # CORRECTED: Use the pickle module to load .pkl files
# with open(term_file, 'rb') as f:
#     term = pickle.load(f)

# with open(preterm_file, 'rb') as f:
#     preterm = pickle.load(f)

# # Ensure they are numpy arrays (in case the pickle contained lists or other structures)
# term = np.array(term)
# preterm = np.array(preterm)

# if term.ndim != 3 or preterm.ndim != 3:
#     raise ValueError(
#         f"Expected arrays with shape [B, N, N]; got {term.shape} and {preterm.shape}"
#     )
# if term.shape[1:] != preterm.shape[1:]:
#     raise ValueError(
#         f"Network shapes do not match: {term.shape[1:]} vs {preterm.shape[1:]}"
#     )

# combined = np.concatenate([preterm, term], axis=0)
# labels = np.concatenate([
#     np.full(len(preterm), "preterm"),
#     np.full(len(term), "term"),
# ])

# np.save(combined_path, combined)
# np.save(labels_path, labels)

# print(f"Preterm: {preterm.shape}")
# print(f"Term:    {term.shape}")
# print(f"Combined: {combined.shape}")
# print(f"Saved matrices: {combined_path}")
# print(f"Saved labels:   {labels_path}")