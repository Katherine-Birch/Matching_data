# import argparse
# import copy
# from pathlib import Path

# import torch
# import config

# from Set_up import run_phase_1
# from Benchmarking import run_phase_2
# from Permutation_scaling import run_phase_3, finetune_model
# from empirical_task import run_phase_4
# from models import (
#     MLPGumbelSinkhorn,
#     GINEncoder,
#     GATEncoder,
#     GraphMatchingModel,
# )

# CHECKPOINT_DIR = Path("checkpoints")
# CHECKPOINT_DIR.mkdir(exist_ok=True)


# def load_payload():
#     path = Path(config.PREPROCESSED_DATA_PATH)
#     if not path.exists():
#         raise FileNotFoundError(
#             f"Missing {path}. Run Phase 1 first: python Main.py --run-phase 1"
#         )
#     return torch.load(path, weights_only=False)


# def build_models():
#     return {
#         "mlp": MLPGumbelSinkhorn(
#             num_nodes=config.NUM_NODES
#         ).to(config.DEVICE),
#         "gin": GraphMatchingModel(
#             GINEncoder(in_channels=5)
#         ).to(config.DEVICE),
#         "gat": GraphMatchingModel(
#             GATEncoder(in_channels=5)
#         ).to(config.DEVICE),
#     }


# def checkpoint_path(name, stage):
#     return CHECKPOINT_DIR / f"{name}_{stage}.pt"


# def save_models(models, stage):
#     for name, model in models.items():
#         torch.save(model.state_dict(), checkpoint_path(name, stage))


# def load_models(stage):
#     models = build_models()
#     for name, model in models.items():
#         path = checkpoint_path(name, stage)
#         if not path.exists():
#             raise FileNotFoundError(
#                 f"Missing checkpoint {path}. Run Phase 3 first."
#             )
#         model.load_state_dict(
#             torch.load(path, map_location=config.DEVICE, weights_only=True)
#         )
#         model.eval()
#     return models


# def train_models(payload):
#     """Train baseline models on seen permutations, then fine-tune copies."""
#     baseline_models = build_models()
#     finetuned_models = {}

#     for name, model in baseline_models.items():
#         print(f"Training baseline {name.upper()} on seen permutations...")
#         baseline_models[name] = finetune_model(
#             model=model,
#             train_graphs=payload["real_graphs"],
#             manifest_path="manifest_seen.pt",
#             model_kind=name,
#             epochs=10,
#             device=config.DEVICE,
#         )

#         finetuned = copy.deepcopy(baseline_models[name])
#         print(f"Fine-tuning {name.upper()} on fine-tuning permutations...")
#         finetuned_models[name] = finetune_model(
#             model=finetuned,
#             train_graphs=payload["real_graphs"],
#             manifest_path="manifest_finetune.pt",
#             model_kind=name,
#             epochs=10,
#             device=config.DEVICE,
#         )

#     save_models(baseline_models, "baseline")
#     save_models(finetuned_models, "finetuned")
#     return baseline_models, finetuned_models


# def main():
#     parser = argparse.ArgumentParser(
#         description="Graph Matching Benchmark Pipeline"
#     )
#     parser.add_argument(
#         "--run-phase", type=int, choices=[1, 2, 3, 4]
#     )
#     parser.add_argument("--run-all", action="store_true")
#     args = parser.parse_args()

#     if not args.run_phase and not args.run_all:
#         parser.print_help()
#         return

#     # Phase 1 only prepares data and manifests. It does not train models.
#     if args.run_phase == 1 or args.run_all:
#         print("Running Phase 1: Data setup...")
#         run_phase_1()
#         if args.run_phase == 1:
#             return

#     payload = load_payload()

#     if args.run_phase == 3 or args.run_all:
#         print("Running Phase 3: Training and permutation scaling...")
#         baseline_models, finetuned_models = train_models(payload)

#         for name in baseline_models:
#             print(f"Evaluating permutation generalisation for {name.upper()}...")
#             run_phase_3(
#                 model=baseline_models[name],
#                 finetuned_model=finetuned_models[name],
#                 graphs=payload["real_graphs"],
#                 model_kind=name,
#                 output_path=f"results/permutation_scaling_{name}.csv",
#                 device=config.DEVICE,
#             )
#     else:
#         baseline_models = load_models("baseline")
#         finetuned_models = load_models("finetuned")

#     if args.run_phase == 2 or args.run_all:
#         print("Running Phase 2: Benchmarking...")
#         run_phase_2(
#             models_dict=baseline_models,
#             empirical_graphs=payload["real_graphs"],
#             synthetic_graphs=payload.get("synthetic_graphs"),
#             output_path="results/benchmark_results.csv",
#             device=config.DEVICE,
#         )

#     if args.run_phase == 4 or args.run_all:
#         print("Running Phase 4: Synthetic-to-empirical alignment...")
#         run_phase_4(
#             model=finetuned_models["mlp"],
#             synthetic_graphs=payload.get("synthetic_graphs"),
#             empirical_graphs=payload["real_graphs"],
#             model_kind="mlp",
#             output_path="results/synthetic_alignment.csv",
#             device=config.DEVICE,
#         )


# if __name__ == "__main__":
#     main()

import argparse
import copy
from pathlib import Path

import torch
import config

from Set_up import run_phase_1
from Benchmarking import run_phase_2
from Permutation_scaling import run_phase_3, finetune_model
from empirical_task import run_phase_4
from models import (
    MLPGumbelSinkhorn,
    GINEncoder,
    GATEncoder,
    GraphMatchingModel,
)
import numpy as np

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


def load_payload():
    path = Path(config.PREPROCESSED_DATA_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run Phase 1 first: python Main.py --run-phase 1"
        )
    return torch.load(path, weights_only=False)


def build_models():
    return {
        "mlp": MLPGumbelSinkhorn(
            num_nodes=config.NUM_NODES
        ).to(config.DEVICE),
        "gin": GraphMatchingModel(
            GINEncoder(in_channels=5)
        ).to(config.DEVICE),
        "gat": GraphMatchingModel(
            GATEncoder(in_channels=5)
        ).to(config.DEVICE),
    }


def checkpoint_path(name, stage):
    return CHECKPOINT_DIR / f"{name}_{stage}.pt"


def save_models(models, stage):
    for name, model in models.items():
        torch.save(model.state_dict(), checkpoint_path(name, stage))


def load_models(stage):
    models = build_models()
    for name, model in models.items():
        path = checkpoint_path(name, stage)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint {path}. Run Phase 3 first."
            )
        model.load_state_dict(
            torch.load(path, map_location=config.DEVICE, weights_only=True)
        )
        model.eval()
    return models


def train_models(payload, epochs=10, force_retrain=False):
    """Train or resume baseline models, then train or resume fine-tuned copies."""
    baseline_models = {}
    finetuned_models = {}
    fresh_models = build_models()

    for name, fresh_model in fresh_models.items():
        baseline_path = checkpoint_path(name, "baseline")
        finetuned_path = checkpoint_path(name, "finetuned")

        if baseline_path.exists() and not force_retrain:
            print(f"Loading existing baseline {name.upper()} checkpoint...")
            baseline = build_models()[name]
            baseline.load_state_dict(
                torch.load(
                    baseline_path,
                    map_location=config.DEVICE,
                    weights_only=True,
                )
            )
            baseline.eval()
        else:
            print(f"Training baseline {name.upper()} on seen permutations...")
            baseline = finetune_model(
                model=fresh_model,
                train_graphs=payload["real_graphs"],
                manifest_path="manifest_seen.pt",
                model_kind=name,
                epochs=epochs,
                device=config.DEVICE,
            )
            torch.save(baseline.state_dict(), baseline_path)

        baseline_models[name] = baseline

        if finetuned_path.exists() and not force_retrain:
            print(f"Loading existing fine-tuned {name.upper()} checkpoint...")
            finetuned = build_models()[name]
            finetuned.load_state_dict(
                torch.load(
                    finetuned_path,
                    map_location=config.DEVICE,
                    weights_only=True,
                )
            )
            finetuned.eval()
        else:
            print(f"Fine-tuning {name.upper()} on fine-tuning permutations...")
            finetuned = copy.deepcopy(baseline)
            finetuned = finetune_model(
                model=finetuned,
                train_graphs=payload["real_graphs"],
                manifest_path="manifest_finetune.pt",
                model_kind=name,
                epochs=epochs,
                device=config.DEVICE,
            )
            torch.save(finetuned.state_dict(), finetuned_path)

        finetuned_models[name] = finetuned

    return baseline_models, finetuned_models


def main():
    parser = argparse.ArgumentParser(
        description="Graph Matching Benchmark Pipeline"
    )
    parser.add_argument(
        "--run-phase", type=int, choices=[1, 2, 3, 4]
    )
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore existing checkpoints and retrain all models.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    if not args.run_phase and not args.run_all:
        parser.print_help()
        return

    if args.run_phase == 1 or args.run_all:
        print("Running Phase 1: Data setup...")
        run_phase_1()
        if args.run_phase == 1:
            return

    payload = load_payload()

    if args.run_phase == 3 or args.run_all:
        print("Running Phase 3: Training and permutation scaling...")
        baseline_models, finetuned_models = train_models(
            payload,
            epochs=args.epochs,
            force_retrain=args.force_retrain,
        )

        for name in baseline_models:
            print(f"Evaluating permutation generalisation for {name.upper()}...")
            run_phase_3(
                model=baseline_models[name],
                finetuned_model=finetuned_models[name],
                graphs=payload["real_graphs"],
                model_kind=name,
                output_path=f"results/permutation_scaling_{name}.csv",
                device=config.DEVICE,
            )
    else:
        baseline_models = load_models("baseline")
        finetuned_models = load_models("finetuned")

    if args.run_phase == 2 or args.run_all:
        print("Running Phase 2: Benchmarking...")
        run_phase_2(
            models_dict=baseline_models,
            empirical_graphs=payload["real_graphs"],
            synthetic_graphs=payload.get("synthetic_graphs"),
            output_path="results/benchmark_results.csv",
            device=config.DEVICE,
        )

    # if args.run_phase == 4 or args.run_all:
    #     print("Running Phase 4: Synthetic-to-empirical alignment...")

    # for model_kind, model in finetuned_models.items():
    #     print(f"Evaluating {model_kind.upper()} alignment...")


    #     run_phase_4(
    #         model=model,
    #         synthetic_graphs=payload.get("synthetic_graphs"),
    #         empirical_graphs=payload["real_graphs"],
    #         model_kind=model_kind,
    #         output_path=f"results/synthetic_alignment_{model_kind}.csv",
    #         device=config.DEVICE,
    #     )
    if args.run_phase == 4 or args.run_all:
        print("Running Phase 4 with selected synthetic-transfer models...")

    split = np.load("synthetic_manifests/graph_split_indices.npz")
    synthetic_test = payload["synthetic_graphs"][split["test"]]

    transfer_models = build_models()
    for model_kind, model in transfer_models.items():
        checkpoint = (
            CHECKPOINT_DIR
            / f"{model_kind}_synthetic_transfer_best.pt"
        )
        model.load_state_dict(
            torch.load(
                checkpoint,
                map_location=config.DEVICE,
                weights_only=True,
            )
        )
        model.eval()

        run_phase_4(
            model=model,
            synthetic_graphs=synthetic_test,
            empirical_graphs=payload["real_graphs"],
            model_kind=model_kind,
            output_path=(
                f"results/synthetic_transfer_alignment_{model_kind}.csv"
            ),
            device=config.DEVICE,
        )


if __name__ == "__main__":
    main()
