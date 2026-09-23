from __future__ import annotations

import argparse
import copy
import csv
import time
from pathlib import Path

import numpy as np
import torch

import config
from models import (
    GATEncoder,
    GINEncoder,
    GraphMatchingModel,
    MLPGumbelSinkhorn,
)
from Permutation_scaling import evaluate_manifest, finetune_model, summarize_rows

CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")
MANIFEST_DIR = Path("synthetic_manifests")

MODEL_ORDER = ("mlp", "gin", "gat")
DEFAULT_EPOCHS = (0, 1, 3, 5, 10, 20)


def build_models():
    return {
        "mlp": MLPGumbelSinkhorn(num_nodes=config.NUM_NODES),
        "gin": GraphMatchingModel(GINEncoder(in_channels=5)),
        "gat": GraphMatchingModel(GATEncoder(in_channels=5)),
    }


def load_empirical_baseline(model_kind, device):
    model = build_models()[model_kind].to(device)
    path = CHECKPOINT_DIR / f"{model_kind}_baseline.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Train the empirical baseline models first."
        )
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.eval()


def split_graphs(graphs, seed=42, train_fraction=0.70, val_fraction=0.15):
    n = len(graphs)
    if n < 3:
        raise ValueError("At least three synthetic networks are required")
    if train_fraction <= 0 or val_fraction <= 0:
        raise ValueError("train_fraction and val_fraction must be positive")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be below one")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train = max(1, int(round(train_fraction * n)))
    n_val = max(1, int(round(val_fraction * n)))
    n_train = min(n_train, n - 2)
    n_val = min(n_val, n - n_train - 1)

    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]

    return (
        graphs[torch.as_tensor(train_idx)],
        graphs[torch.as_tensor(val_idx)],
        graphs[torch.as_tensor(test_idx)],
        {"train": train_idx, "validation": val_idx, "test": test_idx},
    )


def save_manifest(graph_count, seeds, split_name, output_path):
    """Save graph-specific known permutations for a local graph subset."""
    records = []
    for graph_id in range(int(graph_count)):
        for seed in seeds:
            combined_seed = int(seed) + 1_000_003 * int(graph_id)
            generator = torch.Generator().manual_seed(combined_seed)
            perm = torch.randperm(config.NUM_NODES, generator=generator)
            records.append({
                "graph_id": graph_id,
                "seed": int(seed),
                "perm": perm,
                "inverse_perm": torch.argsort(perm),
                "split": split_name,
            })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, output_path)
    return output_path


def mean_summary(rows):
    summary = summarize_rows(rows)
    if len(summary) != 1:
        raise RuntimeError(f"Expected one summary row, received {len(summary)}")
    return summary[0]


def write_csv(rows, path):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fit_from_empirical_baseline(
    model_kind,
    train_graphs,
    train_manifest,
    epochs,
    device,
    seed,
):
    """Reload the same empirical baseline for every epoch condition."""
    model = load_empirical_baseline(model_kind, device)
    if epochs == 0:
        return model, 0.0

    candidate = copy.deepcopy(model)
    start = time.perf_counter()
    candidate = finetune_model(
        model=candidate,
        train_graphs=train_graphs,
        manifest_path=train_manifest,
        model_kind=model_kind,
        epochs=epochs,
        device=device,
        seed=seed,
    )
    training_time = time.perf_counter() - start
    return candidate.eval(), training_time


def run_epoch_sweep(
    synthetic_graphs,
    epochs_grid=DEFAULT_EPOCHS,
    device=None,
    split_seed=42,
    training_seed=0,
):
    device = device or config.DEVICE
    synthetic_graphs = torch.as_tensor(synthetic_graphs).float().cpu()

    train_graphs, val_graphs, test_graphs, indices = split_graphs(
        synthetic_graphs, seed=split_seed
    )

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_manifest = save_manifest(
        len(train_graphs), config.SEEDS_FINETUNE,
        "synthetic_train", MANIFEST_DIR / "train.pt",
    )
    val_manifest = save_manifest(
        len(val_graphs), list(range(20, 25)),
        "synthetic_validation", MANIFEST_DIR / "validation.pt",
    )
    test_manifest = save_manifest(
        len(test_graphs), list(range(25, 30)),
        "synthetic_test", MANIFEST_DIR / "test.pt",
    )
    np.savez(
        MANIFEST_DIR / "graph_split_indices.npz",
        train=indices["train"],
        validation=indices["validation"],
        test=indices["test"],
    )

    sweep_rows = []
    selected = {}

    for model_kind in MODEL_ORDER:
        print(f"\n=== {model_kind.upper()} synthetic transfer sweep ===")
        candidates = []

        for epochs in epochs_grid:
            print(f"Epoch condition: {epochs}")
            model, training_time = fit_from_empirical_baseline(
                model_kind=model_kind,
                train_graphs=train_graphs,
                train_manifest=train_manifest,
                epochs=int(epochs),
                device=device,
                seed=training_seed,
            )

            val_rows = evaluate_manifest(
                model=model,
                graphs=val_graphs,
                manifest_path=val_manifest,
                model_kind=model_kind,
                device=device,
                label="synthetic_validation",
            )
            summary = mean_summary(val_rows)
            row = {
                "model_kind": model_kind,
                "epochs": int(epochs),
                "training_time_seconds": float(training_time),
                "validation_accuracy": summary["mean_accuracy"],
                "validation_l1": summary["mean_l1"],
                "validation_frobenius": summary["mean_frobenius"],
                "validation_forward_time": summary["mean_forward_time"],
                "validation_hungarian_time": summary["mean_hungarian_time"],
                "validation_total_time": summary["mean_total_time"],
            }
            sweep_rows.append(row)
            candidates.append((row, model))

        # Select using validation accuracy, then lower L1 as a tie-breaker.
        best_row, best_model = max(
            candidates,
            key=lambda item: (
                item[0]["validation_accuracy"],
                -item[0]["validation_l1"],
            ),
        )
        selected[model_kind] = best_row

        checkpoint_path = CHECKPOINT_DIR / f"{model_kind}_synthetic_transfer_best.pt"
        torch.save(best_model.state_dict(), checkpoint_path)
        print(
            f"Selected {model_kind.upper()}: {best_row['epochs']} epochs; "
            f"validation accuracy={best_row['validation_accuracy']:.4f}; "
            f"validation L1={best_row['validation_l1']:.6f}"
        )

        test_rows = evaluate_manifest(
            model=best_model,
            graphs=test_graphs,
            manifest_path=test_manifest,
            model_kind=model_kind,
            device=device,
            label="synthetic_test",
        )
        write_csv(
            test_rows,
            RESULTS_DIR / f"synthetic_transfer_test_{model_kind}.csv",
        )
        test_summary = mean_summary(test_rows)
        write_csv(
            [{
                **best_row,
                "test_accuracy": test_summary["mean_accuracy"],
                "test_l1": test_summary["mean_l1"],
                "test_frobenius": test_summary["mean_frobenius"],
                "test_forward_time": test_summary["mean_forward_time"],
                "test_hungarian_time": test_summary["mean_hungarian_time"],
                "test_total_time": test_summary["mean_total_time"],
            }],
            RESULTS_DIR / f"synthetic_transfer_test_{model_kind}_summary.csv",
        )

    write_csv(sweep_rows, RESULTS_DIR / "synthetic_transfer_epoch_sweep.csv")
    write_csv(list(selected.values()), RESULTS_DIR / "synthetic_transfer_selected_epochs.csv")
    return sweep_rows, selected


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune empirical graph matchers on known synthetic permutations"
    )
    parser.add_argument(
        "--epochs", nargs="+", type=int, default=list(DEFAULT_EPOCHS),
        help="Independent epoch conditions, e.g. --epochs 0 1 3 5 10 20",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--training-seed", type=int, default=0)
    args = parser.parse_args()

    payload_path = Path(config.PREPROCESSED_DATA_PATH)
    if not payload_path.exists():
        raise FileNotFoundError(f"Missing {payload_path}. Run Phase 1 first.")
    payload = torch.load(payload_path, weights_only=False)
    synthetic_graphs = payload.get("synthetic_graphs")
    if synthetic_graphs is None:
        raise ValueError("The preprocessed payload contains no synthetic_graphs")

    run_epoch_sweep(
        synthetic_graphs=synthetic_graphs,
        epochs_grid=tuple(args.epochs),
        device=config.DEVICE,
        split_seed=args.split_seed,
        training_seed=args.training_seed,
    )


if __name__ == "__main__":
    main()
