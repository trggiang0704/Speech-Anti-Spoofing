from pathlib import Path

import pandas as pd

import torch

from models.cnn import CNN

from src.dataset import SpeechDataset

from evaluation.verification_eval import (
    verification_evaluate,
)

# ======================================================
# CONFIG
# ======================================================

DATASET_NAME = "VSASV_PAPER_50000"

MODEL_NAME = "CNN"

LABEL_MAP = {

    "bonafide": 0,

    "adversarial_attack": 1,

    "voice_conversion": 2,

}

# ======================================================
# PATHS
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DIR = (

    PROJECT_ROOT

    / "datasets"

    / "logmelspec"

    / DATASET_NAME

)

LABELS_CSV = (

    DATASET_DIR

    / "labels.csv"

)

MODEL_PATH = (

    PROJECT_ROOT

    / "outputs"

    / f"stage1_{MODEL_NAME.lower()}"

    / "best_model.pth"

)

OUTPUT_DIR = (

    PROJECT_ROOT

    / "outputs"

    / "stage2_verification"

)

OUTPUT_DIR.mkdir(

    parents=True,

    exist_ok=True,

)

# ======================================================
# MAIN
# ======================================================

def main():

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )

    print()

    print("=" * 70)

    print(

        "STAGE 2 : SPEAKER VERIFICATION"

    )

    print("=" * 70)

    print(

        f"Device: {device}"

    )

    # ==================================================
    # DATASET
    # ==================================================

    test_dataset = SpeechDataset(

        dataset_dir=DATASET_DIR,

        split="test",

        label_column="label",

        label_map=LABEL_MAP,

    )

    print()

    print(

        f"Test samples: {len(test_dataset)}"

    )

    # ==================================================
    # MODEL
    # ==================================================

    model = CNN(

        num_classes=3,

        embedding_dim=128,

    ).to(device)

    model.load_state_dict(

        torch.load(

            MODEL_PATH,

            map_location=device,

        )

    )

    print()

    print(

        "Loaded best_model.pth"

    )

    # ==================================================
    # VERIFICATION EVALUATION
    # ==================================================

    metrics = verification_evaluate(

        model=model,

        dataset=test_dataset,

        labels_csv=LABELS_CSV,

        device=device,

        output_dir=OUTPUT_DIR,

    )

    # ==================================================
    # DISPLAY RESULT
    # ==================================================

    print()

    print("=" * 70)

    print(

        "FINAL RESULT"

    )

    print("=" * 70)

    for key, value in metrics.items():

        if isinstance(

            value,

            float,

        ):

            print(

                f"{key}: {value:.4f}"

            )

        else:

            print(

                f"{key}: {value}"

            )

    print()

    print(

        "Saved to:"

    )

    print(

        OUTPUT_DIR

    )


if __name__ == "__main__":

    main()