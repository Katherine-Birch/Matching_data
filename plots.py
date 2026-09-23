# # from pathlib import Path

# # import numpy as np
# # import pandas as pd
# # import torch
# # import matplotlib.pyplot as plt
# # import seaborn as sns

# # RESULT_FILES = {
# #     "MLP": Path("results/synthetic_alignment_mlp.csv"),
# #     "GIN": Path("results/synthetic_alignment_gin.csv"),
# #     "GAT": Path("results/synthetic_alignment_gat.csv"),
# # }

# # MAPPING_FILES = {
# #     model: path.with_name(path.stem + "_mappings.pt")
# #     for model, path in RESULT_FILES.items()
# # }

# # SCALING_SUMMARY_FILES = {
# #     "MLP": Path("results/permutation_scaling_mlp_summary.csv"),
# #     "GIN": Path("results/permutation_scaling_gin_summary.csv"),
# #     "GAT": Path("results/permutation_scaling_gat_summary.csv"),
# # }

# # PREPROCESSED_DATA_PATH = Path("preprocessed_data.pt")
# # OUTPUT_DIR = Path("results/figures")
# # OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # MODEL_ORDER = ["MLP", "GIN", "GAT"]
# # PALETTE = {"MLP": "#4C78A8", "GIN": "#F58518", "GAT": "#54A24B"}


# # def benjamini_hochberg(p_values, alpha=0.05):
# #     """Return BH-adjusted p-values and rejection decisions."""
# #     p = np.asarray(p_values, dtype=float)
# #     n = len(p)
# #     order = np.argsort(p)
# #     ranked = p[order]

# #     adjusted_ranked = ranked * n / np.arange(1, n + 1)
# #     adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
# #     adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)

# #     adjusted = np.empty(n, dtype=float)
# #     adjusted[order] = adjusted_ranked
# #     return adjusted <= alpha, adjusted


# # def load_results():
# #     frames = []
# #     for model, path in RESULT_FILES.items():
# #         if not path.exists():
# #             raise FileNotFoundError(f"Missing {path}")
# #         frame = pd.read_csv(path)
# #         frame["model"] = model
# #         frames.append(frame)

# #     data = pd.concat(frames, ignore_index=True)
# #     numeric = [
# #         "alignment_improvement", "alignment_p_value", "frobenius",
# #         "graph_id", "hub_overlap", "hub_overlap_null_mean",
# #         "hub_overlap_p_value", "l1", "random_l1_mean",
# #     ]
# #     for column in numeric:
# #         data[column] = pd.to_numeric(data[column], errors="coerce")

# #     data["alignment_improvement_pct"] = 100.0 * data["alignment_improvement"]
# #     data["hub_enrichment"] = data["hub_overlap"] - data["hub_overlap_null_mean"]
# #     data["error_ratio"] = data["l1"] / data["random_l1_mean"]

# #     data["alignment_p_fdr"] = np.nan
# #     data["hub_p_fdr"] = np.nan
# #     data["alignment_significant_fdr"] = False
# #     data["hub_significant_fdr"] = False

# #     for model in MODEL_ORDER:
# #         mask = data["model"].eq(model)
# #         reject, adjusted = benjamini_hochberg(
# #             data.loc[mask, "alignment_p_value"].to_numpy()
# #         )
# #         data.loc[mask, "alignment_p_fdr"] = adjusted
# #         data.loc[mask, "alignment_significant_fdr"] = reject

# #         reject, adjusted = benjamini_hochberg(
# #             data.loc[mask, "hub_overlap_p_value"].to_numpy()
# #         )
# #         data.loc[mask, "hub_p_fdr"] = adjusted
# #         data.loc[mask, "hub_significant_fdr"] = reject

# #     return data


# # def add_box_and_points(ax, data, x, y):
# #     sns.boxplot(
# #         data=data, x=x, y=y, order=MODEL_ORDER, palette=PALETTE,
# #         width=0.55, showfliers=False, linewidth=1.2, ax=ax,
# #     )
# #     sns.stripplot(
# #         data=data, x=x, y=y, order=MODEL_ORDER, palette=PALETTE,
# #         size=3.2, alpha=0.45, jitter=0.22, ax=ax,
# #     )


# # def make_alignment_figure(data):
# #     sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
# #     fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# #     ax = axes[0, 0]
# #     add_box_and_points(ax, data, "model", "alignment_improvement_pct")
# #     ax.axhline(0, color="black", linestyle="--", linewidth=1)
# #     ax.set_title("A. Alignment improvement over random")
# #     ax.set_xlabel("")
# #     ax.set_ylabel("Improvement in L1 error (%)")

# #     ax = axes[0, 1]
# #     add_box_and_points(ax, data, "model", "hub_overlap")
# #     null_means = data.groupby("model")["hub_overlap_null_mean"].mean()
# #     for position, model in enumerate(MODEL_ORDER):
# #         ax.scatter(
# #             position, null_means.loc[model], marker="D", s=58,
# #             color="black", facecolor="white", linewidth=1.4, zorder=8,
# #             label="Mean random expectation" if position == 0 else None,
# #         )
# #     ax.set_title("B. Hub overlap after alignment")
# #     ax.set_xlabel("")
# #     ax.set_ylabel("Top-10 hub overlap")
# #     ax.set_ylim(-0.03, 1.03)
# #     ax.legend(frameon=True, loc="upper left")

# #     ax = axes[1, 0]
# #     for model in MODEL_ORDER:
# #         subset = data[data["model"].eq(model)]
# #         ax.scatter(
# #             subset["alignment_improvement_pct"], subset["hub_overlap"],
# #             s=34, alpha=0.65, color=PALETTE[model], label=model,
# #         )
# #     ax.axvline(0, color="black", linestyle="--", linewidth=1)
# #     ax.set_title("C. Reconstruction improvement versus hub overlap")
# #     ax.set_xlabel("Improvement in L1 error (%)")
# #     ax.set_ylabel("Top-10 hub overlap")
# #     ax.legend(frameon=True)

# #     ax = axes[1, 1]
# #     significance = []
# #     for model in MODEL_ORDER:
# #         subset = data[data["model"].eq(model)]
# #         significance.append({
# #             "model": model,
# #             "outcome": "Alignment error",
# #             "percentage": 100 * subset["alignment_significant_fdr"].mean(),
# #         })
# #         significance.append({
# #             "model": model,
# #             "outcome": "Hub overlap",
# #             "percentage": 100 * subset["hub_significant_fdr"].mean(),
# #         })
# #     significance = pd.DataFrame(significance)
# #     sns.barplot(
# #         data=significance, x="model", y="percentage", hue="outcome",
# #         order=MODEL_ORDER, palette=["#7A5195", "#EF5675"], ax=ax,
# #     )
# #     ax.set_title("D. Graphs significant after FDR correction")
# #     ax.set_xlabel("")
# #     ax.set_ylabel("Synthetic networks (%)")
# #     ax.set_ylim(0, 105)
# #     ax.legend(title="", frameon=True)

# #     fig.suptitle(
# #         "Synthetic-to-empirical topology-based alignment",
# #         fontsize=16, fontweight="bold", y=0.995,
# #     )
# #     fig.tight_layout()

# #     png_path = OUTPUT_DIR / "synthetic_alignment_comparison.png"
# #     pdf_path = OUTPUT_DIR / "synthetic_alignment_comparison.pdf"
# #     fig.savefig(png_path, dpi=300, bbox_inches="tight")
# #     fig.savefig(pdf_path, bbox_inches="tight")
# #     plt.close(fig)
# #     return png_path, pdf_path


# # def load_mapping_lookup(path):
# #     if not path.exists():
# #         raise FileNotFoundError(
# #             f"Missing {path}. Rerun Phase 4 with the updated empirical_task.py."
# #         )
# #     records = torch.load(path, map_location="cpu", weights_only=False)
# #     return {
# #         int(record["graph_id"]): torch.as_tensor(record["mapping"]).long()
# #         for record in records
# #     }


# # def choose_representative_graph(data):
# #     graph_sets = [
# #         set(data.loc[data["model"].eq(model), "graph_id"].astype(int))
# #         for model in MODEL_ORDER
# #     ]
# #     shared = set.intersection(*graph_sets)
# #     if not shared:
# #         raise ValueError("No graph IDs are shared across all models")

# #     gat = data[
# #         data["model"].eq("GAT") & data["graph_id"].isin(shared)
# #     ].copy()
# #     median_value = gat["alignment_improvement"].median()
# #     index = (gat["alignment_improvement"] - median_value).abs().idxmin()
# #     return int(gat.loc[index, "graph_id"])


# # def make_matrix_figure(data):
# #     if not PREPROCESSED_DATA_PATH.exists():
# #         raise FileNotFoundError(f"Missing {PREPROCESSED_DATA_PATH}")

# #     payload = torch.load(
# #         PREPROCESSED_DATA_PATH, map_location="cpu", weights_only=False
# #     )
# #     empirical = torch.as_tensor(payload["real_graphs"]).float()
# #     synthetic = torch.as_tensor(payload["synthetic_graphs"]).float()
# #     template = empirical.mean(dim=0).clone()
# #     template.fill_diagonal_(0)

# #     graph_id = choose_representative_graph(data)
# #     A_synth = synthetic[graph_id]
# #     mappings = {
# #         model: load_mapping_lookup(MAPPING_FILES[model])[graph_id]
# #         for model in MODEL_ORDER
# #     }
# #     aligned = {
# #         model: A_synth[mapping][:, mapping]
# #         for model, mapping in mappings.items()
# #     }
# #     residual = {
# #         model: (template - matrix).abs()
# #         for model, matrix in aligned.items()
# #     }

# #     matrix_vmax = float(max(template.max(), A_synth.max()))
# #     residual_vmax = float(max(x.max() for x in residual.values()))
# #     stats = data[data["graph_id"].eq(graph_id)].set_index("model")

# #     sns.set_theme(style="white", context="paper", font_scale=1.0)
# #     fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))

# #     panels = [
# #         (axes[0, 0], template, "Empirical mean template", "viridis", matrix_vmax),
# #         (axes[0, 1], A_synth, "Raw synthetic order", "viridis", matrix_vmax),
# #         (axes[0, 2], aligned["MLP"], "MLP aligned", "viridis", matrix_vmax),
# #         (axes[0, 3], aligned["GIN"], "GIN aligned", "viridis", matrix_vmax),
# #         (axes[1, 0], aligned["GAT"], "GAT aligned", "viridis", matrix_vmax),
# #         (axes[1, 1], residual["MLP"], "|Empirical - MLP|", "magma", residual_vmax),
# #         (axes[1, 2], residual["GIN"], "|Empirical - GIN|", "magma", residual_vmax),
# #         (axes[1, 3], residual["GAT"], "|Empirical - GAT|", "magma", residual_vmax),
# #     ]

# #     matrix_image = None
# #     residual_image = None
# #     for ax, matrix, title, cmap, vmax in panels:
# #         image = ax.imshow(matrix.numpy(), cmap=cmap, vmin=0, vmax=vmax)
# #         ax.set_title(title)
# #         ax.set_xticks([])
# #         ax.set_yticks([])
# #         if cmap == "viridis":
# #             matrix_image = image
# #         else:
# #             residual_image = image

# #     for column, model in enumerate(("MLP", "GIN"), start=2):
# #         row = stats.loc[model]
# #         axes[0, column].set_xlabel(
# #             f"L1={row['l1']:.4f}; improvement={100*row['alignment_improvement']:.2f}%"
# #         )
# #     gat_row = stats.loc["GAT"]
# #     axes[1, 0].set_xlabel(
# #         f"L1={gat_row['l1']:.4f}; improvement={100*gat_row['alignment_improvement']:.2f}%"
# #     )

# #     fig.colorbar(
# #         matrix_image, ax=axes[0, :], fraction=0.018, pad=0.02,
# #         label="Normalized edge weight",
# #     )
# #     fig.colorbar(
# #         residual_image, ax=axes[1, 1:], fraction=0.025, pad=0.02,
# #         label="Absolute residual",
# #     )
# #     fig.suptitle(
# #         f"Representative synthetic network alignment (graph {graph_id})",
# #         fontsize=15, fontweight="bold",
# #     )
# #     fig.subplots_adjust(top=0.90, wspace=0.12, hspace=0.28)

# #     png_path = OUTPUT_DIR / "synthetic_alignment_matrices.png"
# #     pdf_path = OUTPUT_DIR / "synthetic_alignment_matrices.pdf"
# #     fig.savefig(png_path, dpi=300, bbox_inches="tight")
# #     fig.savefig(pdf_path, bbox_inches="tight")
# #     plt.close(fig)
# #     return graph_id, png_path, pdf_path


# # def load_efficiency_data():
# #     frames = []
# #     for model, path in SCALING_SUMMARY_FILES.items():
# #         if not path.exists():
# #             raise FileNotFoundError(f"Missing {path}")
# #         frame = pd.read_csv(path)
# #         frame["model"] = model
# #         frames.append(frame)
# #     return pd.concat(frames, ignore_index=True)


# # def make_efficiency_figure(efficiency):
# #     plot_data = efficiency[
# #         efficiency["split"].isin(["baseline_unseen", "finetuned_unseen"])
# #     ].copy()
# #     baseline = plot_data[plot_data["split"].eq("baseline_unseen")].copy()
# #     baseline["model"] = pd.Categorical(
# #         baseline["model"], categories=MODEL_ORDER, ordered=True
# #     )
# #     baseline = baseline.sort_values("model")

# #     sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
# #     fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

# #     ax = axes[0]
# #     x = np.arange(len(baseline))
# #     forward_ms = 1000 * baseline["mean_forward_time"].to_numpy()
# #     hungarian_ms = 1000 * baseline["mean_hungarian_time"].to_numpy()
# #     ax.bar(x, forward_ms, color="#4C78A8", label="Neural forward pass")
# #     ax.bar(
# #         x, hungarian_ms, bottom=forward_ms,
# #         color="#F58518", label="Hungarian assignment",
# #     )
# #     ax.set_xticks(x, baseline["model"].astype(str))
# #     ax.set_ylabel("Mean time per graph-permutation pair (ms)")
# #     ax.set_title("A. Inference-time decomposition")
# #     ax.legend(frameon=True)

# #     ax = axes[1]
# #     markers = {"baseline_unseen": "o", "finetuned_unseen": "s"}
# #     labels = {
# #         "baseline_unseen": "Baseline on unseen",
# #         "finetuned_unseen": "Fine-tuned on unseen",
# #     }
# #     for split, subset in plot_data.groupby("split"):
# #         for _, row in subset.iterrows():
# #             ax.scatter(
# #                 1000 * row["mean_total_time"],
# #                 100 * row["mean_accuracy"],
# #                 color=PALETTE[row["model"]], marker=markers[split],
# #                 s=85, edgecolor="black", linewidth=0.6,
# #                 label=f"{row['model']} - {labels[split]}",
# #             )
# #     ax.set_xscale("log")
# #     ax.set_xlabel("Mean total inference time (ms, log scale)")
# #     ax.set_ylabel("Permutation accuracy (%)")
# #     ax.set_title("B. Accuracy-efficiency trade-off")
# #     handles, legend_labels = ax.get_legend_handles_labels()
# #     unique = dict(zip(legend_labels, handles))
# #     ax.legend(unique.values(), unique.keys(), fontsize=8, frameon=True)

# #     fig.suptitle(
# #         "Computational efficiency of permutation recovery",
# #         fontsize=15, fontweight="bold",
# #     )
# #     fig.tight_layout()

# #     png_path = OUTPUT_DIR / "permutation_efficiency.png"
# #     pdf_path = OUTPUT_DIR / "permutation_efficiency.pdf"
# #     fig.savefig(png_path, dpi=300, bbox_inches="tight")
# #     fig.savefig(pdf_path, bbox_inches="tight")
# #     plt.close(fig)
# #     return png_path, pdf_path


# # def make_summary(data):
# #     rows = []
# #     for model in MODEL_ORDER:
# #         subset = data[data["model"].eq(model)]
# #         rows.append({
# #             "model": model,
# #             "n_graphs": len(subset),
# #             "mean_alignment_improvement": subset["alignment_improvement"].mean(),
# #             "median_alignment_improvement": subset["alignment_improvement"].median(),
# #             "mean_l1": subset["l1"].mean(),
# #             "mean_random_l1": subset["random_l1_mean"].mean(),
# #             "mean_hub_overlap": subset["hub_overlap"].mean(),
# #             "mean_null_hub_overlap": subset["hub_overlap_null_mean"].mean(),
# #             "alignment_fdr_significant_fraction": subset["alignment_significant_fdr"].mean(),
# #             "hub_fdr_significant_fraction": subset["hub_significant_fdr"].mean(),
# #         })

# #     summary = pd.DataFrame(rows)
# #     summary_path = OUTPUT_DIR / "synthetic_alignment_summary.csv"
# #     summary.to_csv(summary_path, index=False)

# #     detailed_path = OUTPUT_DIR / "synthetic_alignment_with_fdr.csv"
# #     data.to_csv(detailed_path, index=False)
# #     return summary, summary_path, detailed_path


# # def report_graph_coverage(data):
# #     graph_sets = {
# #         model: set(data.loc[data["model"].eq(model), "graph_id"].astype(int))
# #         for model in MODEL_ORDER
# #     }
# #     shared = set.intersection(*graph_sets.values())
# #     print("Graph counts:")
# #     for model in MODEL_ORDER:
# #         print(f"  {model}: {len(graph_sets[model])}")
# #     print(f"  Shared graph IDs across all models: {len(shared)}")


# # if __name__ == "__main__":
# #     results = load_results()
# #     report_graph_coverage(results)

# #     alignment_png, alignment_pdf = make_alignment_figure(results)
# #     summary, summary_csv, detailed_csv = make_summary(results)

# #     print("\nAlignment summary:")
# #     print(summary.to_string(index=False))
# #     print(f"\nSaved {alignment_png}")
# #     print(f"Saved {alignment_pdf}")
# #     print(f"Saved {summary_csv}")
# #     print(f"Saved {detailed_csv}")

# #     try:
# #         graph_id, matrix_png, matrix_pdf = make_matrix_figure(results)
# #         print(f"Saved representative graph {graph_id} matrix figure: {matrix_png}")
# #         print(f"Saved {matrix_pdf}")
# #     except FileNotFoundError as error:
# #         print(f"Skipping matrix figure: {error}")

# #     try:
# #         efficiency = load_efficiency_data()
# #         efficiency_png, efficiency_pdf = make_efficiency_figure(efficiency)
# #         print(f"Saved {efficiency_png}")
# #         print(f"Saved {efficiency_pdf}")
# #     except FileNotFoundError as error:
# #         print(f"Skipping efficiency figure: {error}")
# from pathlib import Path

# from pathlib import Path

# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# from matplotlib.patches import Patch
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import torch

# RESULT_FILES = {
#     "MLP": Path("results/synthetic_transfer_alignment_mlp.csv"),
#     "GIN": Path("results/synthetic_transfer_alignment_gin.csv"),
#     "GAT": Path("results/synthetic_transfer_alignment_gat.csv"),
# }

# MAPPING_FILES = {
#     model: path.with_name(path.stem + "_mappings.pt")
#     for model, path in RESULT_FILES.items()
# }

# # SCALING_SUMMARY_FILES = {
# #     "MLP": Path("results/synthetic_transfer_permutation_scaling_mlp_summary.csv"),
# #     "GIN": Path("results/synthetic_transfer_permutation_scaling_gin_summary.csv"),
# #     "GAT": Path("results/synthetic_transfer_permutation_scaling_gat_summary.csv"),
# # }
# SCALING_SUMMARY_FILES = {
#     "MLP": Path("results/synthetic_transfer_test_mlp_summary.csv"),
#     "GIN": Path("results/synthetic_transfer_test_gin_summary.csv"),
#     "GAT": Path("results/synthetic_transfer_test_gat_summary.csv"),
# }

# PREPROCESSED_DATA_PATH = Path("preprocessed_data.pt")
# SYNTHETIC_SPLIT_PATH = Path("synthetic_manifests/graph_split_indices.npz")
# OUTPUT_DIR = Path("results/figures")
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MODEL_ORDER = ["MLP", "GIN", "GAT"]
# PALETTE = {"MLP": "#4C78A8", "GIN": "#F58518", "GAT": "#54A24B"}

# # Used only for the display-scaled matrix figure. Each matrix is divided by
# # its own non-zero 99th percentile and clipped to [0, 1]. Quantitative
# # alignment metrics are always calculated from the original matrices.
# DISPLAY_QUANTILE = 0.99

# # Fraction of all possible undirected node pairs retained in the binary
# # strong-edge comparison. This affects display/overlap analyses only; the
# # original weighted matrices remain unchanged for L1 and null statistics.
# TOP_EDGE_FRACTION = 0.10


# def benjamini_hochberg(p_values, alpha=0.05):
#     """Return BH-adjusted p-values and rejection decisions."""
#     p = np.asarray(p_values, dtype=float)
#     valid = np.isfinite(p)
#     adjusted = np.full(len(p), np.nan, dtype=float)
#     rejected = np.zeros(len(p), dtype=bool)

#     if not valid.any():
#         return rejected, adjusted

#     pv = p[valid]
#     order = np.argsort(pv)
#     ranked = pv[order]
#     n = len(ranked)

#     adjusted_ranked = ranked * n / np.arange(1, n + 1)
#     adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
#     adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)

#     restored = np.empty(n, dtype=float)
#     restored[order] = adjusted_ranked
#     adjusted[valid] = restored
#     rejected[valid] = restored <= alpha
#     return rejected, adjusted


# def load_results():
#     frames = []
#     for model, path in RESULT_FILES.items():
#         if not path.exists():
#             raise FileNotFoundError(f"Missing {path}")
#         frame = pd.read_csv(path)
#         frame["model"] = model
#         frames.append(frame)

#     data = pd.concat(frames, ignore_index=True)
#     numeric = [
#         "alignment_improvement", "alignment_p_value", "frobenius",
#         "graph_id", "hub_overlap", "hub_overlap_null_mean",
#         "hub_overlap_p_value", "l1", "random_l1_mean",
#     ]
#     for column in numeric:
#         if column in data:
#             data[column] = pd.to_numeric(data[column], errors="coerce")

#     data["alignment_improvement_pct"] = 100.0 * data["alignment_improvement"]
#     data["hub_enrichment"] = data["hub_overlap"] - data["hub_overlap_null_mean"]
#     data["error_ratio"] = data["l1"] / data["random_l1_mean"]

#     data["alignment_p_fdr"] = np.nan
#     data["hub_p_fdr"] = np.nan
#     data["alignment_significant_fdr"] = False
#     data["hub_significant_fdr"] = False

#     for model in MODEL_ORDER:
#         mask = data["model"].eq(model)
#         rejected, adjusted = benjamini_hochberg(
#             data.loc[mask, "alignment_p_value"].to_numpy()
#         )
#         data.loc[mask, "alignment_p_fdr"] = adjusted
#         data.loc[mask, "alignment_significant_fdr"] = rejected

#         rejected, adjusted = benjamini_hochberg(
#             data.loc[mask, "hub_overlap_p_value"].to_numpy()
#         )
#         data.loc[mask, "hub_p_fdr"] = adjusted
#         data.loc[mask, "hub_significant_fdr"] = rejected

#     return data


# def add_box_and_points(ax, data, x, y):
#     sns.boxplot(
#         data=data, x=x, y=y, order=MODEL_ORDER, hue=x,
#         palette=PALETTE, legend=False, width=0.55,
#         showfliers=False, linewidth=1.2, ax=ax,
#     )
#     sns.stripplot(
#         data=data, x=x, y=y, order=MODEL_ORDER, hue=x,
#         palette=PALETTE, legend=False, size=3.2,
#         alpha=0.45, jitter=0.22, ax=ax,
#     )


# def make_alignment_figure(data):
#     sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
#     fig, axes = plt.subplots(2, 2, figsize=(12, 9))

#     ax = axes[0, 0]
#     add_box_and_points(ax, data, "model", "alignment_improvement_pct")
#     ax.axhline(0, color="black", linestyle="--", linewidth=1)
#     ax.set_title("A. Alignment improvement over random")
#     ax.set_xlabel("")
#     ax.set_ylabel("Improvement in L1 error (%)")

#     ax = axes[0, 1]
#     add_box_and_points(ax, data, "model", "hub_overlap")
#     null_means = data.groupby("model")["hub_overlap_null_mean"].mean()
#     for position, model in enumerate(MODEL_ORDER):
#         ax.scatter(
#             position, null_means.loc[model], marker="D", s=58,
#             color="black", facecolor="white", linewidth=1.4, zorder=8,
#             label="Mean random expectation" if position == 0 else None,
#         )
#     ax.set_title("B. Hub overlap after alignment")
#     ax.set_xlabel("")
#     ax.set_ylabel("Top-10 hub overlap")
#     ax.set_ylim(-0.03, 1.03)
#     ax.legend(frameon=True, loc="upper left")

#     ax = axes[1, 0]
#     for model in MODEL_ORDER:
#         subset = data[data["model"].eq(model)]
#         ax.scatter(
#             subset["alignment_improvement_pct"], subset["hub_overlap"],
#             s=34, alpha=0.65, color=PALETTE[model], label=model,
#         )
#     ax.axvline(0, color="black", linestyle="--", linewidth=1)
#     ax.set_title("C. Reconstruction improvement versus hub overlap")
#     ax.set_xlabel("Improvement in L1 error (%)")
#     ax.set_ylabel("Top-10 hub overlap")
#     ax.legend(frameon=True)

#     significance = []
#     for model in MODEL_ORDER:
#         subset = data[data["model"].eq(model)]
#         significance.extend([
#             {
#                 "model": model,
#                 "outcome": "Alignment error",
#                 "percentage": 100 * subset["alignment_significant_fdr"].mean(),
#             },
#             {
#                 "model": model,
#                 "outcome": "Hub overlap",
#                 "percentage": 100 * subset["hub_significant_fdr"].mean(),
#             },
#         ])

#     ax = axes[1, 1]
#     significance = pd.DataFrame(significance)
#     sns.barplot(
#         data=significance, x="model", y="percentage", hue="outcome",
#         order=MODEL_ORDER, palette=["#7A5195", "#EF5675"], ax=ax,
#     )
#     ax.set_title("D. Graphs significant after FDR correction")
#     ax.set_xlabel("")
#     ax.set_ylabel("Synthetic networks (%)")
#     ax.set_ylim(0, 105)
#     ax.legend(title="", frameon=True)

#     fig.suptitle(
#         "Synthetic-to-empirical topology-based alignment",
#         fontsize=16, fontweight="bold", y=0.995,
#     )
#     fig.tight_layout()

#     png_path = OUTPUT_DIR / "synthetic_alignment_comparison.png"
#     pdf_path = OUTPUT_DIR / "synthetic_alignment_comparison.pdf"
#     fig.savefig(png_path, dpi=300, bbox_inches="tight")
#     fig.savefig(pdf_path, bbox_inches="tight")
#     plt.close(fig)
#     return png_path, pdf_path


# def load_mapping_lookup(path):
#     if not path.exists():
#         raise FileNotFoundError(
#             f"Missing {path}. Rerun Phase 4 with mapping saving enabled."
#         )
#     records = torch.load(path, map_location="cpu", weights_only=False)
#     return {
#         int(record["graph_id"]): torch.as_tensor(record["mapping"]).long()
#         for record in records
#     }


# def choose_representative_graph(data):
#     """Choose the shared graph closest to median GAT improvement."""
#     graph_sets = [
#         set(data.loc[data["model"].eq(model), "graph_id"].astype(int))
#         for model in MODEL_ORDER
#     ]
#     shared = set.intersection(*graph_sets)
#     if not shared:
#         raise ValueError("No graph IDs are shared across all models")

#     gat = data[
#         data["model"].eq("GAT") & data["graph_id"].isin(shared)
#     ].copy()
#     median_value = gat["alignment_improvement"].median()
#     index = (gat["alignment_improvement"] - median_value).abs().idxmin()
#     return int(gat.loc[index, "graph_id"])


# def upper_triangle_l1(A, B):
#     n = A.shape[0]
#     idx = torch.triu_indices(n, n, offset=1)
#     return float((A[idx[0], idx[1]] - B[idx[0], idx[1]]).abs().mean())


# def scale_for_display(matrix, quantile=DISPLAY_QUANTILE):
#     """Independently rescale a copy for plotting only."""
#     matrix = torch.as_tensor(matrix).detach().cpu().float().clone()
#     nonzero = matrix[matrix > 0]
#     if nonzero.numel() == 0:
#         return matrix

#     scale = torch.quantile(nonzero, quantile)
#     if not torch.isfinite(scale) or scale <= 0:
#         scale = nonzero.max()
#     if scale > 0:
#         matrix = (matrix / scale).clamp(0, 1)
#     return matrix


# def load_matrix_inputs(data):
#     if not PREPROCESSED_DATA_PATH.exists():
#         raise FileNotFoundError(f"Missing {PREPROCESSED_DATA_PATH}")

#     payload = torch.load(
#         PREPROCESSED_DATA_PATH, map_location="cpu", weights_only=False
#     )
#     empirical = torch.as_tensor(payload["real_graphs"]).float()

#     split = np.load(SYNTHETIC_SPLIT_PATH)
#     test_indices = torch.as_tensor(split["test"], dtype=torch.long)
#     synthetic = torch.as_tensor(payload["synthetic_graphs"]).float()[test_indices]
    
#     # synthetic = torch.as_tensor(payload["synthetic_graphs"]).float()
#     template = empirical.mean(dim=0).clone()
#     template.fill_diagonal_(0)

#     graph_id = choose_representative_graph(data)
#     A_synth = synthetic[graph_id]
#     mappings = {
#         model: load_mapping_lookup(MAPPING_FILES[model])[graph_id]
#         for model in MODEL_ORDER
#     }
#     aligned = {
#         model: A_synth[mapping][:, mapping]
#         for model, mapping in mappings.items()
#     }
#     stats = data[data["graph_id"].eq(graph_id)].set_index("model")
#     return graph_id, template, A_synth, aligned, stats


# def matrix_annotation(l1, improvement):
#     return f"L1={l1:.4f}; improvement={100 * improvement:.2f}%"


# def make_matrix_figure(data, display_scaled=False):
#     graph_id, template, A_synth, aligned, stats = load_matrix_inputs(data)

#     raw_l1 = upper_triangle_l1(template, A_synth)
#     random_mean = float(stats["random_l1_mean"].mean())
#     raw_improvement = (random_mean - raw_l1) / (random_mean + 1e-12)

#     if display_scaled:
#         plotted_template = scale_for_display(template)
#         plotted_raw = scale_for_display(A_synth)
#         plotted_aligned = {
#             model: scale_for_display(matrix)
#             for model, matrix in aligned.items()
#         }
#         matrix_vmax = 1.0
#         scale_note = (
#             f"Display only: each matrix independently scaled by its non-zero "
#             f"{100 * DISPLAY_QUANTILE:.0f}th percentile"
#         )
#         suffix = "display_scaled"
#     else:
#         plotted_template = template
#         plotted_raw = A_synth
#         plotted_aligned = aligned
#         matrix_vmax = float(max(template.max(), A_synth.max()))
#         scale_note = "Common colour scale; original preprocessed matrices"
#         suffix = "common_scale"

#     residual = {
#         model: (plotted_template - matrix).abs()
#         for model, matrix in plotted_aligned.items()
#     }
#     residual_vmax = float(max(x.max() for x in residual.values()))

#     sns.set_theme(style="white", context="paper", font_scale=1.0)
#     fig, axes = plt.subplots(2, 4, figsize=(15, 7.8))

#     panels = [
#         (axes[0, 0], plotted_template, "Empirical mean template", "viridis", matrix_vmax),
#         (axes[0, 1], plotted_raw, "Raw synthetic order", "viridis", matrix_vmax),
#         (axes[0, 2], plotted_aligned["MLP"], "MLP aligned", "viridis", matrix_vmax),
#         (axes[0, 3], plotted_aligned["GIN"], "GIN aligned", "viridis", matrix_vmax),
#         (axes[1, 0], plotted_aligned["GAT"], "GAT aligned", "viridis", matrix_vmax),
#         (axes[1, 1], residual["MLP"], "|Empirical - MLP|", "magma", residual_vmax),
#         (axes[1, 2], residual["GIN"], "|Empirical - GIN|", "magma", residual_vmax),
#         (axes[1, 3], residual["GAT"], "|Empirical - GAT|", "magma", residual_vmax),
#     ]

#     matrix_image = None
#     residual_image = None
#     for ax, matrix, title, cmap, vmax in panels:
#         image = ax.imshow(matrix.numpy(), cmap=cmap, vmin=0, vmax=vmax)
#         ax.set_title(title)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         if cmap == "viridis":
#             matrix_image = image
#         else:
#             residual_image = image

#     axes[0, 1].set_xlabel(matrix_annotation(raw_l1, raw_improvement))
#     for column, model in enumerate(("MLP", "GIN"), start=2):
#         row = stats.loc[model]
#         axes[0, column].set_xlabel(
#             matrix_annotation(row["l1"], row["alignment_improvement"])
#         )
#     gat_row = stats.loc["GAT"]
#     axes[1, 0].set_xlabel(
#         matrix_annotation(gat_row["l1"], gat_row["alignment_improvement"])
#     )

#     fig.colorbar(
#         matrix_image, ax=axes[0, :], fraction=0.018, pad=0.02,
#         label="Display-scaled edge weight" if display_scaled else "Normalized edge weight",
#     )
#     fig.colorbar(
#         residual_image, ax=axes[1, 1:], fraction=0.025, pad=0.02,
#         label="Display-scaled absolute residual" if display_scaled else "Absolute residual",
#     )

#     fig.suptitle(
#         f"Representative topology-based alignment (graph {graph_id})",
#         fontsize=15, fontweight="bold", y=0.98,
#     )
#     fig.text(
#         0.5, 0.015,
#         scale_note + ". Reported L1 and improvement values use original matrices.",
#         ha="center", fontsize=9,
#     )
#     fig.subplots_adjust(top=0.90, bottom=0.08, wspace=0.12, hspace=0.30)

#     png_path = OUTPUT_DIR / f"synthetic_alignment_matrices_{suffix}.png"
#     pdf_path = OUTPUT_DIR / f"synthetic_alignment_matrices_{suffix}.pdf"
#     fig.savefig(png_path, dpi=300, bbox_inches="tight")
#     fig.savefig(pdf_path, bbox_inches="tight")
#     plt.close(fig)
#     return graph_id, png_path, pdf_path


# def rank_transform_weights(matrix, tol=1e-12):
#     """Rank non-zero undirected edge weights within one matrix to (0, 1]."""
#     matrix = torch.as_tensor(matrix).detach().cpu().float()
#     n = matrix.shape[0]
#     idx = torch.triu_indices(n, n, offset=1)
#     values = matrix[idx[0], idx[1]].numpy()
#     nonzero = np.abs(values) > tol

#     ranked_values = np.zeros_like(values, dtype=float)
#     if nonzero.any():
#         ranked_values[nonzero] = (
#             pd.Series(values[nonzero]).rank(method="average", pct=True).to_numpy()
#         )

#     ranked = torch.zeros((n, n), dtype=torch.float32)
#     ranked[idx[0], idx[1]] = torch.from_numpy(ranked_values).float()
#     ranked = ranked + ranked.T
#     return ranked


# def make_rank_matrix_figure(data):
#     """Plot within-network edge-weight ranks without changing analysis values."""
#     graph_id, template, A_synth, aligned, stats = load_matrix_inputs(data)

#     ranked_template = rank_transform_weights(template)
#     ranked_raw = rank_transform_weights(A_synth)
#     ranked_aligned = {
#         model: rank_transform_weights(matrix)
#         for model, matrix in aligned.items()
#     }
#     residual = {
#         model: (ranked_template - matrix).abs()
#         for model, matrix in ranked_aligned.items()
#     }

#     raw_l1 = upper_triangle_l1(template, A_synth)
#     random_mean = float(stats["random_l1_mean"].mean())
#     raw_improvement = (random_mean - raw_l1) / (random_mean + 1e-12)

#     sns.set_theme(style="white", context="paper", font_scale=1.0)
#     fig, axes = plt.subplots(2, 4, figsize=(15, 7.8))

#     panels = [
#         (axes[0, 0], ranked_template, "Empirical mean template", "viridis"),
#         (axes[0, 1], ranked_raw, "Raw synthetic order", "viridis"),
#         (axes[0, 2], ranked_aligned["MLP"], "MLP aligned", "viridis"),
#         (axes[0, 3], ranked_aligned["GIN"], "GIN aligned", "viridis"),
#         (axes[1, 0], ranked_aligned["GAT"], "GAT aligned", "viridis"),
#         (axes[1, 1], residual["MLP"], "|Empirical rank - MLP rank|", "magma"),
#         (axes[1, 2], residual["GIN"], "|Empirical rank - GIN rank|", "magma"),
#         (axes[1, 3], residual["GAT"], "|Empirical rank - GAT rank|", "magma"),
#     ]

#     matrix_image = None
#     residual_image = None
#     for ax, matrix, title, cmap in panels:
#         image = ax.imshow(matrix.numpy(), cmap=cmap, vmin=0, vmax=1)
#         ax.set_title(title)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         if cmap == "viridis":
#             matrix_image = image
#         else:
#             residual_image = image

#     axes[0, 1].set_xlabel(matrix_annotation(raw_l1, raw_improvement))
#     for column, model in enumerate(("MLP", "GIN"), start=2):
#         row = stats.loc[model]
#         axes[0, column].set_xlabel(
#             matrix_annotation(row["l1"], row["alignment_improvement"])
#         )
#     gat_row = stats.loc["GAT"]
#     axes[1, 0].set_xlabel(
#         matrix_annotation(gat_row["l1"], gat_row["alignment_improvement"])
#     )

#     fig.colorbar(
#         matrix_image, ax=axes[0, :], fraction=0.018, pad=0.02,
#         label="Within-network non-zero edge-weight percentile",
#     )
#     fig.colorbar(
#         residual_image, ax=axes[1, 1:], fraction=0.025, pad=0.02,
#         label="Absolute difference in edge-weight percentile",
#     )

#     fig.suptitle(
#         f"Rank-transformed topology-based alignment (graph {graph_id})",
#         fontsize=15, fontweight="bold", y=0.98,
#     )
#     fig.text(
#         0.5, 0.015,
#         "Display only: non-zero edge weights ranked separately within each matrix; "
#         "zero edges remain zero. Reported L1 and improvement use original matrices.",
#         ha="center", fontsize=9,
#     )
#     fig.subplots_adjust(top=0.90, bottom=0.08, wspace=0.12, hspace=0.30)

#     png_path = OUTPUT_DIR / "synthetic_alignment_matrices_rank.png"
#     pdf_path = OUTPUT_DIR / "synthetic_alignment_matrices_rank.pdf"
#     fig.savefig(png_path, dpi=300, bbox_inches="tight")
#     fig.savefig(pdf_path, bbox_inches="tight")
#     plt.close(fig)
#     return graph_id, png_path, pdf_path


# def top_edge_mask(matrix, fraction=TOP_EDGE_FRACTION, tol=1e-12):
#     """Return a symmetric mask containing the strongest edge fraction."""
#     matrix = torch.as_tensor(matrix).detach().cpu().float()
#     if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
#         raise ValueError("matrix must have shape [N, N]")
#     if not 0 < fraction <= 1:
#         raise ValueError("fraction must be in (0, 1]")

#     n = matrix.shape[0]
#     idx = torch.triu_indices(n, n, offset=1)
#     values = matrix[idx[0], idx[1]]
#     positive = torch.where(values > tol)[0]

#     requested = max(1, int(round(fraction * values.numel())))
#     retained = min(requested, positive.numel())

#     mask = torch.zeros((n, n), dtype=torch.bool)
#     if retained == 0:
#         return mask

#     selected_local = torch.topk(values[positive], k=retained).indices
#     selected = positive[selected_local]
#     rows, cols = idx[0][selected], idx[1][selected]
#     mask[rows, cols] = True
#     mask[cols, rows] = True
#     return mask


# def strong_edge_overlap_metrics(empirical_mask, candidate_mask):
#     """Calculate upper-triangle overlap metrics for two binary edge masks."""
#     empirical_mask = torch.as_tensor(empirical_mask).bool().cpu()
#     candidate_mask = torch.as_tensor(candidate_mask).bool().cpu()
#     if empirical_mask.shape != candidate_mask.shape:
#         raise ValueError("Masks must have the same shape")

#     n = empirical_mask.shape[0]
#     idx = torch.triu_indices(n, n, offset=1)
#     empirical = empirical_mask[idx[0], idx[1]]
#     candidate = candidate_mask[idx[0], idx[1]]

#     intersection = int((empirical & candidate).sum())
#     union = int((empirical | candidate).sum())
#     n_empirical = int(empirical.sum())
#     n_candidate = int(candidate.sum())

#     precision = intersection / n_candidate if n_candidate else float("nan")
#     recall = intersection / n_empirical if n_empirical else float("nan")
#     f1 = (
#         2 * precision * recall / (precision + recall)
#         if np.isfinite(precision + recall) and precision + recall > 0
#         else float("nan")
#     )
#     jaccard = intersection / union if union else float("nan")

#     return {
#         "empirical_edges": n_empirical,
#         "candidate_edges": n_candidate,
#         "shared_edges": intersection,
#         "jaccard": float(jaccard),
#         "precision": float(precision),
#         "recall": float(recall),
#         "f1": float(f1),
#     }


# def overlap_code_matrix(empirical_mask, candidate_mask):
#     """Encode absent/empirical-only/candidate-only/shared edges as 0/1/2/3."""
#     empirical_mask = torch.as_tensor(empirical_mask).bool().cpu()
#     candidate_mask = torch.as_tensor(candidate_mask).bool().cpu()
#     code = empirical_mask.to(torch.uint8) + 2 * candidate_mask.to(torch.uint8)
#     code.fill_diagonal_(0)
#     return code


# def make_top_edge_figure(data, fraction=TOP_EDGE_FRACTION):
#     """Visualise strongest-edge topology and overlap for one representative graph."""
#     graph_id, template, A_synth, aligned, _ = load_matrix_inputs(data)

#     empirical_mask = top_edge_mask(template, fraction=fraction)
#     raw_mask = top_edge_mask(A_synth, fraction=fraction)
#     aligned_masks = {
#         model: top_edge_mask(matrix, fraction=fraction)
#         for model, matrix in aligned.items()
#     }

#     raw_metrics = strong_edge_overlap_metrics(empirical_mask, raw_mask)
#     model_metrics = {
#         model: strong_edge_overlap_metrics(empirical_mask, mask)
#         for model, mask in aligned_masks.items()
#     }

#     overlap_cmap = ListedColormap([
#         "#FFFFFF", "#4C78A8", "#F58518", "#54A24B",
#     ])
#     binary_cmap = ListedColormap(["#FFFFFF", "#111111"])

#     sns.set_theme(style="white", context="paper", font_scale=1.0)
#     fig, axes = plt.subplots(2, 4, figsize=(15, 7.8))

#     binary_panels = [
#         (axes[0, 0], empirical_mask, "Empirical top edges", None),
#         (axes[0, 1], raw_mask, "Raw synthetic top edges", raw_metrics),
#         (axes[0, 2], aligned_masks["MLP"], "MLP-aligned top edges", model_metrics["MLP"]),
#         (axes[0, 3], aligned_masks["GIN"], "GIN-aligned top edges", model_metrics["GIN"]),
#         (axes[1, 0], aligned_masks["GAT"], "GAT-aligned top edges", model_metrics["GAT"]),
#     ]

#     for ax, mask, title, metrics in binary_panels:
#         ax.imshow(mask.numpy(), cmap=binary_cmap, vmin=0, vmax=1, interpolation="nearest")
#         ax.set_title(title)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         if metrics is not None:
#             ax.set_xlabel(
#                 f"Jaccard={metrics['jaccard']:.3f}; "
#                 f"F1={metrics['f1']:.3f}; shared={metrics['shared_edges']}"
#             )

#     for column, model in enumerate(MODEL_ORDER, start=1):
#         ax = axes[1, column]
#         code = overlap_code_matrix(empirical_mask, aligned_masks[model])
#         ax.imshow(code.numpy(), cmap=overlap_cmap, vmin=0, vmax=3, interpolation="nearest")
#         metrics = model_metrics[model]
#         ax.set_title(f"Empirical versus {model}")
#         ax.set_xlabel(
#             f"Precision={metrics['precision']:.3f}; recall={metrics['recall']:.3f}"
#         )
#         ax.set_xticks([])
#         ax.set_yticks([])

#     legend_handles = [
#         Patch(facecolor="#4C78A8", label="Empirical only"),
#         Patch(facecolor="#F58518", label="Synthetic only"),
#         Patch(facecolor="#54A24B", label="Shared"),
#     ]
#     fig.legend(
#         handles=legend_handles, loc="lower center", ncol=3,
#         frameon=True, bbox_to_anchor=(0.66, 0.015),
#     )

#     percentage = 100 * fraction
#     fig.suptitle(
#         f"Strong-edge topology after alignment (top {percentage:g}%; graph {graph_id})",
#         fontsize=15, fontweight="bold", y=0.98,
#     )
#     fig.text(
#         0.28, 0.02,
#         "Each mask retains the strongest fixed proportion of all possible "
#         "undirected node pairs. Original weighted matrices are unchanged.",
#         ha="center", fontsize=8.5,
#     )
#     fig.subplots_adjust(top=0.90, bottom=0.10, wspace=0.12, hspace=0.30)

#     label = f"top{percentage:g}".replace(".", "p")
#     png_path = OUTPUT_DIR / f"synthetic_alignment_{label}_edges.png"
#     pdf_path = OUTPUT_DIR / f"synthetic_alignment_{label}_edges.pdf"
#     fig.savefig(png_path, dpi=300, bbox_inches="tight")
#     fig.savefig(pdf_path, bbox_inches="tight")
#     plt.close(fig)
#     return graph_id, png_path, pdf_path, raw_metrics, model_metrics


# def summarize_top_edge_overlap(data, fraction=TOP_EDGE_FRACTION):
#     """Calculate strongest-edge overlap for every graph shared across models."""
#     if not PREPROCESSED_DATA_PATH.exists():
#         raise FileNotFoundError(f"Missing {PREPROCESSED_DATA_PATH}")

#     payload = torch.load(
#         PREPROCESSED_DATA_PATH, map_location="cpu", weights_only=False
#     )
#     empirical = torch.as_tensor(payload["real_graphs"]).float()
#     split = np.load(SYNTHETIC_SPLIT_PATH)
#     test_indices = torch.as_tensor(split["test"], dtype=torch.long)
#     synthetic = torch.as_tensor(payload["synthetic_graphs"]).float()[test_indices]
#     # synthetic = torch.as_tensor(payload["synthetic_graphs"]).float()
#     template = empirical.mean(dim=0).clone()
#     template.fill_diagonal_(0)
#     empirical_mask = top_edge_mask(template, fraction=fraction)

#     mapping_lookups = {
#         model: load_mapping_lookup(MAPPING_FILES[model])
#         for model in MODEL_ORDER
#     }
#     graph_sets = [
#         set(data.loc[data["model"].eq(model), "graph_id"].astype(int))
#         & set(mapping_lookups[model])
#         for model in MODEL_ORDER
#     ]
#     shared_graphs = sorted(set.intersection(*graph_sets))

#     rows = []
#     for graph_id in shared_graphs:
#         if graph_id < 0 or graph_id >= len(synthetic):
#             continue
#         A_synth = synthetic[graph_id]

#         raw_metrics = strong_edge_overlap_metrics(
#             empirical_mask, top_edge_mask(A_synth, fraction=fraction)
#         )
#         rows.append({
#             "graph_id": graph_id,
#             "model": "Raw",
#             "top_edge_fraction": fraction,
#             **raw_metrics,
#         })

#         for model in MODEL_ORDER:
#             mapping = mapping_lookups[model][graph_id]
#             aligned = A_synth[mapping][:, mapping]
#             metrics = strong_edge_overlap_metrics(
#                 empirical_mask, top_edge_mask(aligned, fraction=fraction)
#             )
#             rows.append({
#                 "graph_id": graph_id,
#                 "model": model,
#                 "top_edge_fraction": fraction,
#                 **metrics,
#             })

#     summary = pd.DataFrame(rows)
#     percentage = 100 * fraction
#     label = f"top{percentage:g}".replace(".", "p")
#     csv_path = OUTPUT_DIR / f"synthetic_alignment_{label}_edge_overlap.csv"
#     summary.to_csv(csv_path, index=False)
#     return summary, csv_path


# def load_efficiency_data():
#     frames = []
#     for model, path in SCALING_SUMMARY_FILES.items():
#         if not path.exists():
#             raise FileNotFoundError(f"Missing {path}")
#         frame = pd.read_csv(path)
#         frame["model"] = model
#         frames.append(frame)
#     return pd.concat(frames, ignore_index=True)


# def make_efficiency_figure(data):
#     data = data.copy()
#     for column in (
#         "mean_accuracy", "mean_forward_time", "mean_hungarian_time",
#         "mean_total_time",
#     ):
#         data[column] = pd.to_numeric(data[column], errors="coerce")

#     unseen = data[data["split"].isin(["baseline_unseen", "finetuned_unseen"])].copy()
#     unseen["accuracy_pct"] = 100 * unseen["mean_accuracy"]
#     unseen["total_ms"] = 1000 * unseen["mean_total_time"]

#     baseline = unseen[unseen["split"].eq("baseline_unseen")].set_index("model")
#     forward_ms = 1000 * baseline.loc[MODEL_ORDER, "mean_forward_time"]
#     hungarian_ms = 1000 * baseline.loc[MODEL_ORDER, "mean_hungarian_time"]

#     sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
#     fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

#     axes[0].bar(MODEL_ORDER, forward_ms, color="#4C78A8", label="Neural forward pass")
#     axes[0].bar(
#         MODEL_ORDER, hungarian_ms, bottom=forward_ms,
#         color="#F58518", label="Hungarian assignment",
#     )
#     axes[0].set_title("A. Inference-time decomposition")
#     axes[0].set_ylabel("Mean time per graph-permutation pair (ms)")
#     axes[0].legend(frameon=True)

#     markers = {"baseline_unseen": "o", "finetuned_unseen": "s"}
#     labels = {
#         "baseline_unseen": "Baseline on unseen",
#         "finetuned_unseen": "Fine-tuned on unseen",
#     }
#     for _, row in unseen.iterrows():
#         axes[1].scatter(
#             row["total_ms"], row["accuracy_pct"],
#             s=95, marker=markers[row["split"]],
#             color=PALETTE[row["model"]], edgecolor="black",
#             label=f"{row['model']} - {labels[row['split']]}",
#         )
#     axes[1].set_xscale("log")
#     axes[1].set_title("B. Accuracy-efficiency trade-off")
#     axes[1].set_xlabel("Mean total inference time (ms, log scale)")
#     axes[1].set_ylabel("Permutation accuracy (%)")
#     axes[1].set_ylim(-3, 104)
#     axes[1].legend(frameon=True, fontsize=9, loc="upper left")

#     fig.suptitle(
#         "Computational efficiency of permutation recovery",
#         fontsize=16, fontweight="bold", y=1.02,
#     )
#     fig.tight_layout()

#     png_path = OUTPUT_DIR / "permutation_efficiency.png"
#     pdf_path = OUTPUT_DIR / "permutation_efficiency.pdf"
#     fig.savefig(png_path, dpi=300, bbox_inches="tight")
#     fig.savefig(pdf_path, bbox_inches="tight")
#     plt.close(fig)
#     return png_path, pdf_path


# def make_summary(data):
#     rows = []
#     for model in MODEL_ORDER:
#         subset = data[data["model"].eq(model)]
#         rows.append({
#             "model": model,
#             "n_graphs": len(subset),
#             "mean_alignment_improvement_pct": subset["alignment_improvement_pct"].mean(),
#             "median_alignment_improvement_pct": subset["alignment_improvement_pct"].median(),
#             "mean_hub_overlap": subset["hub_overlap"].mean(),
#             "mean_hub_null": subset["hub_overlap_null_mean"].mean(),
#             "alignment_fdr_significant_pct": 100 * subset["alignment_significant_fdr"].mean(),
#             "hub_fdr_significant_pct": 100 * subset["hub_significant_fdr"].mean(),
#         })

#     summary = pd.DataFrame(rows)
#     summary_csv = OUTPUT_DIR / "synthetic_alignment_summary.csv"
#     detailed_csv = OUTPUT_DIR / "synthetic_alignment_with_fdr.csv"
#     summary.to_csv(summary_csv, index=False)
#     data.to_csv(detailed_csv, index=False)
#     return summary, summary_csv, detailed_csv


# def report_graph_coverage(data):
#     sets = {
#         model: set(data.loc[data["model"].eq(model), "graph_id"].astype(int))
#         for model in MODEL_ORDER
#     }
#     shared = set.intersection(*sets.values())
#     print("Graph coverage:")
#     for model in MODEL_ORDER:
#         print(f"  {model}: {len(sets[model])} graphs")
#     print(f"  Shared across all models: {len(shared)} graphs")


# def main():
#     results = load_results()
#     report_graph_coverage(results)

#     alignment_png, alignment_pdf = make_alignment_figure(results)
#     summary, summary_csv, detailed_csv = make_summary(results)

#     print("\nAlignment summary:")
#     print(summary.to_string(index=False))
#     print(f"\nSaved {alignment_png}")
#     print(f"Saved {alignment_pdf}")
#     print(f"Saved {summary_csv}")
#     print(f"Saved {detailed_csv}")

#     try:
#         graph_id, common_png, common_pdf = make_matrix_figure(
#             results, display_scaled=False
#         )
#         _, scaled_png, scaled_pdf = make_matrix_figure(
#             results, display_scaled=True
#         )
#         print(f"Saved representative graph {graph_id} common-scale figure: {common_png}")
#         print(f"Saved {common_pdf}")
#         print(f"Saved display-scaled figure: {scaled_png}")
#         print(f"Saved {scaled_pdf}")
#         _, rank_png, rank_pdf = make_rank_matrix_figure(results)
#         print(f"Saved rank-transformed figure: {rank_png}")
#         print(f"Saved {rank_pdf}")

#         _, top_png, top_pdf, raw_top, model_top = make_top_edge_figure(results)
#         print(f"Saved strongest-edge figure: {top_png}")
#         print(f"Saved {top_pdf}")
#         print(
#             "Representative strongest-edge Jaccard: "
#             + ", ".join(
#                 [f"Raw={raw_top['jaccard']:.3f}"]
#                 + [f"{model}={model_top[model]['jaccard']:.3f}" for model in MODEL_ORDER]
#             )
#         )

#         top_summary, top_csv = summarize_top_edge_overlap(results)
#         print(f"Saved strongest-edge overlap table: {top_csv}")
#         if not top_summary.empty:
#             print("\nMean strongest-edge overlap by method:")
#             print(
#                 top_summary.groupby("model")[["jaccard", "precision", "recall", "f1"]]
#                 .mean()
#                 .round(4)
#                 .to_string()
#             )
#     except (FileNotFoundError, KeyError, ValueError) as error:
#         print(f"Skipping matrix figures: {error}")

#     try:
#         efficiency = load_efficiency_data()
#         efficiency_png, efficiency_pdf = make_efficiency_figure(efficiency)
#         print(f"Saved {efficiency_png}")
#         print(f"Saved {efficiency_pdf}")
#     except FileNotFoundError as error:
#         print(f"Skipping efficiency figure: {error}")


# if __name__ == "__main__":
#     main()

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
import torch

RESULT_FILES = {
    "MLP": Path("results/synthetic_alignment_mlp.csv"),
    "GIN": Path("results/synthetic_alignment_gin.csv"),
    "GAT": Path("results/synthetic_alignment_gat.csv"),
}

MAPPING_FILES = {
    model: path.with_name(path.stem + "_mappings.pt")
    for model, path in RESULT_FILES.items()
}

SCALING_SUMMARY_FILES = {
    "MLP": Path("results/synthetic_transfer_test_mlp_summary.csv"),
    "GIN": Path("results/synthetic_transfer_test_gin_summary.csv"),
    "GAT": Path("results/synthetic_transfer_test_gat_summary.csv"),
}

PREPROCESSED_DATA_PATH = Path("preprocessed_data.pt")
OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["MLP", "GIN", "GAT"]
PALETTE = {"MLP": "#4C78A8", "GIN": "#F58518", "GAT": "#54A24B"}

# Used only for the display-scaled matrix figure. Each matrix is divided by
# its own non-zero 99th percentile and clipped to [0, 1]. Quantitative
# alignment metrics are always calculated from the original matrices.
DISPLAY_QUANTILE = 0.99

# Fraction of all possible undirected node pairs retained in the binary
# strong-edge comparison. This affects display/overlap analyses only; the
# original weighted matrices remain unchanged for L1 and null statistics.
TOP_EDGE_FRACTION = 0.10


def benjamini_hochberg(p_values, alpha=0.05):
    """Return BH-adjusted p-values and rejection decisions."""
    p = np.asarray(p_values, dtype=float)
    valid = np.isfinite(p)
    adjusted = np.full(len(p), np.nan, dtype=float)
    rejected = np.zeros(len(p), dtype=bool)

    if not valid.any():
        return rejected, adjusted

    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)

    adjusted_ranked = ranked * n / np.arange(1, n + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)

    restored = np.empty(n, dtype=float)
    restored[order] = adjusted_ranked
    adjusted[valid] = restored
    rejected[valid] = restored <= alpha
    return rejected, adjusted


def load_results():
    frames = []
    for model, path in RESULT_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        frame = pd.read_csv(path)
        frame["model"] = model
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    numeric = [
        "alignment_improvement", "alignment_p_value", "frobenius",
        "graph_id", "hub_overlap", "hub_overlap_null_mean",
        "hub_overlap_p_value", "l1", "random_l1_mean",
    ]
    for column in numeric:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data["alignment_improvement_pct"] = 100.0 * data["alignment_improvement"]
    data["hub_enrichment"] = data["hub_overlap"] - data["hub_overlap_null_mean"]
    data["error_ratio"] = data["l1"] / data["random_l1_mean"]

    data["alignment_p_fdr"] = np.nan
    data["hub_p_fdr"] = np.nan
    data["alignment_significant_fdr"] = False
    data["hub_significant_fdr"] = False

    for model in MODEL_ORDER:
        mask = data["model"].eq(model)
        rejected, adjusted = benjamini_hochberg(
            data.loc[mask, "alignment_p_value"].to_numpy()
        )
        data.loc[mask, "alignment_p_fdr"] = adjusted
        data.loc[mask, "alignment_significant_fdr"] = rejected

        rejected, adjusted = benjamini_hochberg(
            data.loc[mask, "hub_overlap_p_value"].to_numpy()
        )
        data.loc[mask, "hub_p_fdr"] = adjusted
        data.loc[mask, "hub_significant_fdr"] = rejected

    return data


def add_box_and_points(ax, data, x, y):
    sns.boxplot(
        data=data, x=x, y=y, order=MODEL_ORDER, hue=x,
        palette=PALETTE, legend=False, width=0.55,
        showfliers=False, linewidth=1.2, ax=ax,
    )
    sns.stripplot(
        data=data, x=x, y=y, order=MODEL_ORDER, hue=x,
        palette=PALETTE, legend=False, size=3.2,
        alpha=0.45, jitter=0.22, ax=ax,
    )


def make_alignment_figure(data):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    add_box_and_points(ax, data, "model", "alignment_improvement_pct")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("A. Alignment improvement over random")
    ax.set_xlabel("")
    ax.set_ylabel("Improvement in L1 error (%)")

    ax = axes[0, 1]
    add_box_and_points(ax, data, "model", "hub_overlap")
    null_means = data.groupby("model")["hub_overlap_null_mean"].mean()
    for position, model in enumerate(MODEL_ORDER):
        ax.scatter(
            position, null_means.loc[model], marker="D", s=58,
            color="black", facecolor="white", linewidth=1.4, zorder=8,
            label="Mean random expectation" if position == 0 else None,
        )
    ax.set_title("B. Hub overlap after alignment")
    ax.set_xlabel("")
    ax.set_ylabel("Top-10 hub overlap")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=True, loc="upper left")

    ax = axes[1, 0]
    for model in MODEL_ORDER:
        subset = data[data["model"].eq(model)]
        ax.scatter(
            subset["alignment_improvement_pct"], subset["hub_overlap"],
            s=34, alpha=0.65, color=PALETTE[model], label=model,
        )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("C. Reconstruction improvement versus hub overlap")
    ax.set_xlabel("Improvement in L1 error (%)")
    ax.set_ylabel("Top-10 hub overlap")
    ax.legend(frameon=True)

    significance = []
    for model in MODEL_ORDER:
        subset = data[data["model"].eq(model)]
        significance.extend([
            {
                "model": model,
                "outcome": "Alignment error",
                "percentage": 100 * subset["alignment_significant_fdr"].mean(),
            },
            {
                "model": model,
                "outcome": "Hub overlap",
                "percentage": 100 * subset["hub_significant_fdr"].mean(),
            },
        ])

    ax = axes[1, 1]
    significance = pd.DataFrame(significance)
    sns.barplot(
        data=significance, x="model", y="percentage", hue="outcome",
        order=MODEL_ORDER, palette=["#7A5195", "#EF5675"], ax=ax,
    )
    ax.set_title("D. Graphs significant after FDR correction")
    ax.set_xlabel("")
    ax.set_ylabel("Synthetic networks (%)")
    ax.set_ylim(0, 105)
    ax.legend(title="", frameon=True)

    fig.suptitle(
        "Synthetic-to-empirical topology-based alignment",
        fontsize=16, fontweight="bold", y=0.995,
    )
    fig.tight_layout()

    png_path = OUTPUT_DIR / "synthetic_alignment_comparison.png"
    pdf_path = OUTPUT_DIR / "synthetic_alignment_comparison.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def load_mapping_lookup(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Rerun Phase 4 with mapping saving enabled."
        )
    records = torch.load(path, map_location="cpu", weights_only=False)
    return {
        int(record["graph_id"]): torch.as_tensor(record["mapping"]).long()
        for record in records
    }


def choose_representative_graph(data):
    """Choose the shared graph closest to median GAT improvement."""
    graph_sets = [
        set(data.loc[data["model"].eq(model), "graph_id"].astype(int))
        for model in MODEL_ORDER
    ]
    shared = set.intersection(*graph_sets)
    if not shared:
        raise ValueError("No graph IDs are shared across all models")

    gat = data[
        data["model"].eq("GAT") & data["graph_id"].isin(shared)
    ].copy()
    median_value = gat["alignment_improvement"].median()
    index = (gat["alignment_improvement"] - median_value).abs().idxmin()
    return int(gat.loc[index, "graph_id"])


def upper_triangle_l1(A, B):
    n = A.shape[0]
    idx = torch.triu_indices(n, n, offset=1)
    return float((A[idx[0], idx[1]] - B[idx[0], idx[1]]).abs().mean())


def scale_for_display(matrix, quantile=DISPLAY_QUANTILE):
    """Independently rescale a copy for plotting only."""
    matrix = torch.as_tensor(matrix).detach().cpu().float().clone()
    nonzero = matrix[matrix > 0]
    if nonzero.numel() == 0:
        return matrix

    scale = torch.quantile(nonzero, quantile)
    if not torch.isfinite(scale) or scale <= 0:
        scale = nonzero.max()
    if scale > 0:
        matrix = (matrix / scale).clamp(0, 1)
    return matrix


def load_matrix_inputs(data):
    if not PREPROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {PREPROCESSED_DATA_PATH}")

    payload = torch.load(
        PREPROCESSED_DATA_PATH, map_location="cpu", weights_only=False
    )
    empirical = torch.as_tensor(payload["real_graphs"]).float()
    synthetic = torch.as_tensor(payload["synthetic_graphs"]).float()
    template = empirical.mean(dim=0).clone()
    template.fill_diagonal_(0)

    graph_id = choose_representative_graph(data)
    A_synth = synthetic[graph_id]
    mappings = {
        model: load_mapping_lookup(MAPPING_FILES[model])[graph_id]
        for model in MODEL_ORDER
    }
    aligned = {
        model: A_synth[mapping][:, mapping]
        for model, mapping in mappings.items()
    }
    stats = data[data["graph_id"].eq(graph_id)].set_index("model")
    return graph_id, template, A_synth, aligned, stats


def matrix_annotation(l1, improvement):
    return f"L1={l1:.4f}; improvement={100 * improvement:.2f}%"


def make_matrix_figure(data, display_scaled=False):
    graph_id, template, A_synth, aligned, stats = load_matrix_inputs(data)

    raw_l1 = upper_triangle_l1(template, A_synth)
    random_mean = float(stats["random_l1_mean"].mean())
    raw_improvement = (random_mean - raw_l1) / (random_mean + 1e-12)

    if display_scaled:
        plotted_template = scale_for_display(template)
        plotted_raw = scale_for_display(A_synth)
        plotted_aligned = {
            model: scale_for_display(matrix)
            for model, matrix in aligned.items()
        }
        matrix_vmax = 1.0
        scale_note = (
            f"Display only: each matrix independently scaled by its non-zero "
            f"{100 * DISPLAY_QUANTILE:.0f}th percentile"
        )
        suffix = "display_scaled"
    else:
        plotted_template = template
        plotted_raw = A_synth
        plotted_aligned = aligned
        matrix_vmax = float(max(template.max(), A_synth.max()))
        scale_note = "Common colour scale; original preprocessed matrices"
        suffix = "common_scale"

    residual = {
        model: (plotted_template - matrix).abs()
        for model, matrix in plotted_aligned.items()
    }
    residual_vmax = float(max(x.max() for x in residual.values()))

    sns.set_theme(style="white", context="paper", font_scale=1.0)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.8))

    panels = [
        (axes[0, 0], plotted_template, "Empirical mean template", "viridis", matrix_vmax),
        (axes[0, 1], plotted_raw, "Raw synthetic order", "viridis", matrix_vmax),
        (axes[0, 2], plotted_aligned["MLP"], "MLP aligned", "viridis", matrix_vmax),
        (axes[0, 3], plotted_aligned["GIN"], "GIN aligned", "viridis", matrix_vmax),
        (axes[1, 0], plotted_aligned["GAT"], "GAT aligned", "viridis", matrix_vmax),
        (axes[1, 1], residual["MLP"], "|Empirical - MLP|", "magma", residual_vmax),
        (axes[1, 2], residual["GIN"], "|Empirical - GIN|", "magma", residual_vmax),
        (axes[1, 3], residual["GAT"], "|Empirical - GAT|", "magma", residual_vmax),
    ]

    matrix_image = None
    residual_image = None
    for ax, matrix, title, cmap, vmax in panels:
        image = ax.imshow(matrix.numpy(), cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        if cmap == "viridis":
            matrix_image = image
        else:
            residual_image = image

    axes[0, 1].set_xlabel(matrix_annotation(raw_l1, raw_improvement))
    for column, model in enumerate(("MLP", "GIN"), start=2):
        row = stats.loc[model]
        axes[0, column].set_xlabel(
            matrix_annotation(row["l1"], row["alignment_improvement"])
        )
    gat_row = stats.loc["GAT"]
    axes[1, 0].set_xlabel(
        matrix_annotation(gat_row["l1"], gat_row["alignment_improvement"])
    )

    fig.colorbar(
        matrix_image, ax=axes[0, :], fraction=0.018, pad=0.02,
        label="Display-scaled edge weight" if display_scaled else "Normalized edge weight",
    )
    fig.colorbar(
        residual_image, ax=axes[1, 1:], fraction=0.025, pad=0.02,
        label="Display-scaled absolute residual" if display_scaled else "Absolute residual",
    )

    fig.suptitle(
        f"Representative topology-based alignment (graph {graph_id})",
        fontsize=15, fontweight="bold", y=0.98,
    )
    fig.text(
        0.5, 0.015,
        scale_note + ". Reported L1 and improvement values use original matrices.",
        ha="center", fontsize=9,
    )
    fig.subplots_adjust(top=0.90, bottom=0.08, wspace=0.12, hspace=0.30)

    png_path = OUTPUT_DIR / f"synthetic_alignment_matrices_{suffix}.png"
    pdf_path = OUTPUT_DIR / f"synthetic_alignment_matrices_{suffix}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return graph_id, png_path, pdf_path


def rank_transform_weights(matrix, tol=1e-12):
    """Rank non-zero undirected edge weights within one matrix to (0, 1]."""
    matrix = torch.as_tensor(matrix).detach().cpu().float()
    n = matrix.shape[0]
    idx = torch.triu_indices(n, n, offset=1)
    values = matrix[idx[0], idx[1]].numpy()
    nonzero = np.abs(values) > tol

    ranked_values = np.zeros_like(values, dtype=float)
    if nonzero.any():
        ranked_values[nonzero] = (
            pd.Series(values[nonzero]).rank(method="average", pct=True).to_numpy()
        )

    ranked = torch.zeros((n, n), dtype=torch.float32)
    ranked[idx[0], idx[1]] = torch.from_numpy(ranked_values).float()
    ranked = ranked + ranked.T
    return ranked


def make_rank_matrix_figure(data):
    """Plot within-network edge-weight ranks without changing analysis values."""
    graph_id, template, A_synth, aligned, stats = load_matrix_inputs(data)

    ranked_template = rank_transform_weights(template)
    ranked_raw = rank_transform_weights(A_synth)
    ranked_aligned = {
        model: rank_transform_weights(matrix)
        for model, matrix in aligned.items()
    }
    residual = {
        model: (ranked_template - matrix).abs()
        for model, matrix in ranked_aligned.items()
    }

    raw_l1 = upper_triangle_l1(template, A_synth)
    random_mean = float(stats["random_l1_mean"].mean())
    raw_improvement = (random_mean - raw_l1) / (random_mean + 1e-12)

    sns.set_theme(style="white", context="paper", font_scale=1.0)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.8))

    panels = [
        (axes[0, 0], ranked_template, "Empirical mean template", "viridis"),
        (axes[0, 1], ranked_raw, "Raw synthetic order", "viridis"),
        (axes[0, 2], ranked_aligned["MLP"], "MLP aligned", "viridis"),
        (axes[0, 3], ranked_aligned["GIN"], "GIN aligned", "viridis"),
        (axes[1, 0], ranked_aligned["GAT"], "GAT aligned", "viridis"),
        (axes[1, 1], residual["MLP"], "|Empirical rank - MLP rank|", "magma"),
        (axes[1, 2], residual["GIN"], "|Empirical rank - GIN rank|", "magma"),
        (axes[1, 3], residual["GAT"], "|Empirical rank - GAT rank|", "magma"),
    ]

    matrix_image = None
    residual_image = None
    for ax, matrix, title, cmap in panels:
        image = ax.imshow(matrix.numpy(), cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        if cmap == "viridis":
            matrix_image = image
        else:
            residual_image = image

    axes[0, 1].set_xlabel(matrix_annotation(raw_l1, raw_improvement))
    for column, model in enumerate(("MLP", "GIN"), start=2):
        row = stats.loc[model]
        axes[0, column].set_xlabel(
            matrix_annotation(row["l1"], row["alignment_improvement"])
        )
    gat_row = stats.loc["GAT"]
    axes[1, 0].set_xlabel(
        matrix_annotation(gat_row["l1"], gat_row["alignment_improvement"])
    )

    fig.colorbar(
        matrix_image, ax=axes[0, :], fraction=0.018, pad=0.02,
        label="Within-network non-zero edge-weight percentile",
    )
    fig.colorbar(
        residual_image, ax=axes[1, 1:], fraction=0.025, pad=0.02,
        label="Absolute difference in edge-weight percentile",
    )

    fig.suptitle(
        f"Rank-transformed topology-based alignment (graph {graph_id})",
        fontsize=15, fontweight="bold", y=0.98,
    )
    fig.text(
        0.5, 0.015,
        "Display only: non-zero edge weights ranked separately within each matrix; "
        "zero edges remain zero. Reported L1 and improvement use original matrices.",
        ha="center", fontsize=9,
    )
    fig.subplots_adjust(top=0.90, bottom=0.08, wspace=0.12, hspace=0.30)

    png_path = OUTPUT_DIR / "synthetic_alignment_matrices_rank.png"
    pdf_path = OUTPUT_DIR / "synthetic_alignment_matrices_rank.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return graph_id, png_path, pdf_path


def top_edge_mask(matrix, fraction=TOP_EDGE_FRACTION, tol=1e-12):
    """Return a symmetric mask containing the strongest edge fraction."""
    matrix = torch.as_tensor(matrix).detach().cpu().float()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must have shape [N, N]")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")

    n = matrix.shape[0]
    idx = torch.triu_indices(n, n, offset=1)
    values = matrix[idx[0], idx[1]]
    positive = torch.where(values > tol)[0]

    requested = max(1, int(round(fraction * values.numel())))
    retained = min(requested, positive.numel())

    mask = torch.zeros((n, n), dtype=torch.bool)
    if retained == 0:
        return mask

    selected_local = torch.topk(values[positive], k=retained).indices
    selected = positive[selected_local]
    rows, cols = idx[0][selected], idx[1][selected]
    mask[rows, cols] = True
    mask[cols, rows] = True
    return mask


def strong_edge_overlap_metrics(empirical_mask, candidate_mask):
    """Calculate upper-triangle overlap metrics for two binary edge masks."""
    empirical_mask = torch.as_tensor(empirical_mask).bool().cpu()
    candidate_mask = torch.as_tensor(candidate_mask).bool().cpu()
    if empirical_mask.shape != candidate_mask.shape:
        raise ValueError("Masks must have the same shape")

    n = empirical_mask.shape[0]
    idx = torch.triu_indices(n, n, offset=1)
    empirical = empirical_mask[idx[0], idx[1]]
    candidate = candidate_mask[idx[0], idx[1]]

    intersection = int((empirical & candidate).sum())
    union = int((empirical | candidate).sum())
    n_empirical = int(empirical.sum())
    n_candidate = int(candidate.sum())

    precision = intersection / n_candidate if n_candidate else float("nan")
    recall = intersection / n_empirical if n_empirical else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision + recall) and precision + recall > 0
        else float("nan")
    )
    jaccard = intersection / union if union else float("nan")

    return {
        "empirical_edges": n_empirical,
        "candidate_edges": n_candidate,
        "shared_edges": intersection,
        "jaccard": float(jaccard),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def overlap_code_matrix(empirical_mask, candidate_mask):
    """Encode absent/empirical-only/candidate-only/shared edges as 0/1/2/3."""
    empirical_mask = torch.as_tensor(empirical_mask).bool().cpu()
    candidate_mask = torch.as_tensor(candidate_mask).bool().cpu()
    code = empirical_mask.to(torch.uint8) + 2 * candidate_mask.to(torch.uint8)
    code.fill_diagonal_(0)
    return code


def make_top_edge_figure(data, fraction=TOP_EDGE_FRACTION):
    """Visualise strongest-edge topology and overlap for one representative graph."""
    graph_id, template, A_synth, aligned, _ = load_matrix_inputs(data)

    empirical_mask = top_edge_mask(template, fraction=fraction)
    raw_mask = top_edge_mask(A_synth, fraction=fraction)
    aligned_masks = {
        model: top_edge_mask(matrix, fraction=fraction)
        for model, matrix in aligned.items()
    }

    raw_metrics = strong_edge_overlap_metrics(empirical_mask, raw_mask)
    model_metrics = {
        model: strong_edge_overlap_metrics(empirical_mask, mask)
        for model, mask in aligned_masks.items()
    }

    overlap_cmap = ListedColormap([
        "#FFFFFF", "#4C78A8", "#F58518", "#54A24B",
    ])
    binary_cmap = ListedColormap(["#FFFFFF", "#111111"])

    sns.set_theme(style="white", context="paper", font_scale=1.0)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.8))

    binary_panels = [
        (axes[0, 0], empirical_mask, "Empirical top edges", None),
        (axes[0, 1], raw_mask, "Raw synthetic top edges", raw_metrics),
        (axes[0, 2], aligned_masks["MLP"], "MLP-aligned top edges", model_metrics["MLP"]),
        (axes[0, 3], aligned_masks["GIN"], "GIN-aligned top edges", model_metrics["GIN"]),
        (axes[1, 0], aligned_masks["GAT"], "GAT-aligned top edges", model_metrics["GAT"]),
    ]

    for ax, mask, title, metrics in binary_panels:
        ax.imshow(mask.numpy(), cmap=binary_cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        if metrics is not None:
            ax.set_xlabel(
                f"Jaccard={metrics['jaccard']:.3f}; "
                f"F1={metrics['f1']:.3f}; shared={metrics['shared_edges']}"
            )

    for column, model in enumerate(MODEL_ORDER, start=1):
        ax = axes[1, column]
        code = overlap_code_matrix(empirical_mask, aligned_masks[model])
        ax.imshow(code.numpy(), cmap=overlap_cmap, vmin=0, vmax=3, interpolation="nearest")
        metrics = model_metrics[model]
        ax.set_title(f"Empirical versus {model}")
        ax.set_xlabel(
            f"Precision={metrics['precision']:.3f}; recall={metrics['recall']:.3f}"
        )
        ax.set_xticks([])
        ax.set_yticks([])

    legend_handles = [
        Patch(facecolor="#4C78A8", label="Empirical only"),
        Patch(facecolor="#F58518", label="Synthetic only"),
        Patch(facecolor="#54A24B", label="Shared"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=3,
        frameon=True, bbox_to_anchor=(0.66, 0.015),
    )

    percentage = 100 * fraction
    fig.suptitle(
        f"Strong-edge topology after alignment (top {percentage:g}%; graph {graph_id})",
        fontsize=15, fontweight="bold", y=0.98,
    )
    fig.text(
        0.28, 0.02,
        "Each mask retains the strongest fixed proportion of all possible "
        "undirected node pairs. Original weighted matrices are unchanged.",
        ha="center", fontsize=8.5,
    )
    fig.subplots_adjust(top=0.90, bottom=0.10, wspace=0.12, hspace=0.30)

    label = f"top{percentage:g}".replace(".", "p")
    png_path = OUTPUT_DIR / f"synthetic_alignment_{label}_edges.png"
    pdf_path = OUTPUT_DIR / f"synthetic_alignment_{label}_edges.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return graph_id, png_path, pdf_path, raw_metrics, model_metrics


def summarize_top_edge_overlap(data, fraction=TOP_EDGE_FRACTION):
    """Calculate strongest-edge overlap for every graph shared across models."""
    if not PREPROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {PREPROCESSED_DATA_PATH}")

    payload = torch.load(
        PREPROCESSED_DATA_PATH, map_location="cpu", weights_only=False
    )
    empirical = torch.as_tensor(payload["real_graphs"]).float()
    synthetic = torch.as_tensor(payload["synthetic_graphs"]).float()
    template = empirical.mean(dim=0).clone()
    template.fill_diagonal_(0)
    empirical_mask = top_edge_mask(template, fraction=fraction)

    mapping_lookups = {
        model: load_mapping_lookup(MAPPING_FILES[model])
        for model in MODEL_ORDER
    }
    graph_sets = [
        set(data.loc[data["model"].eq(model), "graph_id"].astype(int))
        & set(mapping_lookups[model])
        for model in MODEL_ORDER
    ]
    shared_graphs = sorted(set.intersection(*graph_sets))

    rows = []
    for graph_id in shared_graphs:
        if graph_id < 0 or graph_id >= len(synthetic):
            continue
        A_synth = synthetic[graph_id]

        raw_metrics = strong_edge_overlap_metrics(
            empirical_mask, top_edge_mask(A_synth, fraction=fraction)
        )
        rows.append({
            "graph_id": graph_id,
            "model": "Raw",
            "top_edge_fraction": fraction,
            **raw_metrics,
        })

        for model in MODEL_ORDER:
            mapping = mapping_lookups[model][graph_id]
            aligned = A_synth[mapping][:, mapping]
            metrics = strong_edge_overlap_metrics(
                empirical_mask, top_edge_mask(aligned, fraction=fraction)
            )
            rows.append({
                "graph_id": graph_id,
                "model": model,
                "top_edge_fraction": fraction,
                **metrics,
            })

    summary = pd.DataFrame(rows)
    percentage = 100 * fraction
    label = f"top{percentage:g}".replace(".", "p")
    csv_path = OUTPUT_DIR / f"synthetic_alignment_{label}_edge_overlap.csv"
    summary.to_csv(csv_path, index=False)
    return summary, csv_path


def load_efficiency_data():
    """Load selected synthetic-transfer test summaries into one common schema."""
    frames = []
    rename = {
        "test_accuracy": "mean_accuracy",
        "test_forward_time": "mean_forward_time",
        "test_hungarian_time": "mean_hungarian_time",
        "test_total_time": "mean_total_time",
        "test_l1": "mean_l1",
        "test_frobenius": "mean_frobenius",
    }
    required = {
        "mean_accuracy", "mean_forward_time",
        "mean_hungarian_time", "mean_total_time",
    }

    for model, path in SCALING_SUMMARY_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")

        frame = pd.read_csv(path).rename(columns=rename)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{path} is missing columns {sorted(missing)}. "
                f"Available columns: {list(frame.columns)}"
            )

        frame["model"] = model
        frame["split"] = "synthetic_transfer_test"
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def make_efficiency_figure(data):
    """Plot timing and accuracy for the selected transfer model of each architecture."""
    data = data.copy()
    numeric = [
        "mean_accuracy", "mean_forward_time",
        "mean_hungarian_time", "mean_total_time",
    ]
    if "epochs" in data:
        numeric.append("epochs")

    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    selected = (
        data.sort_values("model")
        .drop_duplicates("model", keep="last")
        .set_index("model")
        .loc[MODEL_ORDER]
        .reset_index()
    )
    selected["accuracy_pct"] = 100.0 * selected["mean_accuracy"]
    selected["forward_ms"] = 1000.0 * selected["mean_forward_time"]
    selected["hungarian_ms"] = 1000.0 * selected["mean_hungarian_time"]
    selected["total_ms"] = 1000.0 * selected["mean_total_time"]

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].bar(
        selected["model"], selected["forward_ms"],
        color="#4C78A8", label="Neural forward pass",
    )
    axes[0].bar(
        selected["model"], selected["hungarian_ms"],
        bottom=selected["forward_ms"],
        color="#F58518", label="Hungarian assignment",
    )
    axes[0].set_title("A. Inference-time decomposition")
    axes[0].set_ylabel("Mean time per graph-permutation pair (ms)")
    axes[0].legend(frameon=True)

    for _, row in selected.iterrows():
        epoch_label = (
            f"; {int(row['epochs'])} epochs"
            if "epochs" in selected and np.isfinite(row.get("epochs", np.nan))
            else ""
        )
        axes[1].scatter(
            row["total_ms"], row["accuracy_pct"],
            s=110, color=PALETTE[row["model"]],
            edgecolor="black", label=f"{row['model']}{epoch_label}",
        )

    axes[1].set_xscale("log")
    axes[1].set_title("B. Synthetic-transfer accuracy-efficiency trade-off")
    axes[1].set_xlabel("Mean total inference time (ms, log scale)")
    axes[1].set_ylabel("Held-out synthetic permutation accuracy (%)")
    axes[1].set_ylim(-3, 104)
    axes[1].legend(frameon=True, fontsize=9, loc="best")

    fig.suptitle(
        "Computational efficiency after synthetic-domain transfer",
        fontsize=16, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    png_path = OUTPUT_DIR / "synthetic_transfer_efficiency.png"
    pdf_path = OUTPUT_DIR / "synthetic_transfer_efficiency.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def make_summary(data):
    rows = []
    for model in MODEL_ORDER:
        subset = data[data["model"].eq(model)]
        rows.append({
            "model": model,
            "n_graphs": len(subset),
            "mean_alignment_improvement_pct": subset["alignment_improvement_pct"].mean(),
            "median_alignment_improvement_pct": subset["alignment_improvement_pct"].median(),
            "mean_hub_overlap": subset["hub_overlap"].mean(),
            "mean_hub_null": subset["hub_overlap_null_mean"].mean(),
            "alignment_fdr_significant_pct": 100 * subset["alignment_significant_fdr"].mean(),
            "hub_fdr_significant_pct": 100 * subset["hub_significant_fdr"].mean(),
        })

    summary = pd.DataFrame(rows)
    summary_csv = OUTPUT_DIR / "synthetic_alignment_summary.csv"
    detailed_csv = OUTPUT_DIR / "synthetic_alignment_with_fdr.csv"
    summary.to_csv(summary_csv, index=False)
    data.to_csv(detailed_csv, index=False)
    return summary, summary_csv, detailed_csv


def report_graph_coverage(data):
    sets = {
        model: set(data.loc[data["model"].eq(model), "graph_id"].astype(int))
        for model in MODEL_ORDER
    }
    shared = set.intersection(*sets.values())
    print("Graph coverage:")
    for model in MODEL_ORDER:
        print(f"  {model}: {len(sets[model])} graphs")
    print(f"  Shared across all models: {len(shared)} graphs")


def main():
    results = load_results()
    report_graph_coverage(results)

    alignment_png, alignment_pdf = make_alignment_figure(results)
    summary, summary_csv, detailed_csv = make_summary(results)

    print("\nAlignment summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved {alignment_png}")
    print(f"Saved {alignment_pdf}")
    print(f"Saved {summary_csv}")
    print(f"Saved {detailed_csv}")

    try:
        graph_id, common_png, common_pdf = make_matrix_figure(
            results, display_scaled=False
        )
        _, scaled_png, scaled_pdf = make_matrix_figure(
            results, display_scaled=True
        )
        print(f"Saved representative graph {graph_id} common-scale figure: {common_png}")
        print(f"Saved {common_pdf}")
        print(f"Saved display-scaled figure: {scaled_png}")
        print(f"Saved {scaled_pdf}")
        _, rank_png, rank_pdf = make_rank_matrix_figure(results)
        print(f"Saved rank-transformed figure: {rank_png}")
        print(f"Saved {rank_pdf}")

        _, top_png, top_pdf, raw_top, model_top = make_top_edge_figure(results)
        print(f"Saved strongest-edge figure: {top_png}")
        print(f"Saved {top_pdf}")
        print(
            "Representative strongest-edge Jaccard: "
            + ", ".join(
                [f"Raw={raw_top['jaccard']:.3f}"]
                + [f"{model}={model_top[model]['jaccard']:.3f}" for model in MODEL_ORDER]
            )
        )

        top_summary, top_csv = summarize_top_edge_overlap(results)
        print(f"Saved strongest-edge overlap table: {top_csv}")
        if not top_summary.empty:
            print("\nMean strongest-edge overlap by method:")
            print(
                top_summary.groupby("model")[["jaccard", "precision", "recall", "f1"]]
                .mean()
                .round(4)
                .to_string()
            )
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"Skipping matrix figures: {error}")

    try:
        efficiency = load_efficiency_data()
        efficiency_png, efficiency_pdf = make_efficiency_figure(efficiency)
        print(f"Saved {efficiency_png}")
        print(f"Saved {efficiency_pdf}")
    except FileNotFoundError as error:
        print(f"Skipping efficiency figure: {error}")


if __name__ == "__main__":
    main()
