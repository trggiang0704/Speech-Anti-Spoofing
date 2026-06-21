print()

print(

    ">>> TRAIN_STAGE1 VERSION 2 <<<"

)

print()
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from sklearn.metrics import f1_score

from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from tqdm import tqdm

from src.dataset import (

    SpeechDataset,

    debug_dataset,

)
from src.evaluation import evaluate

from models.cnn import CNN

# ======================================================
# CONFIG
# ======================================================

BATCH_SIZE = 64

EPOCHS = 25

PATIENCE = 5

LEARNING_RATE = 1e-3

NUM_WORKERS = 4

LABEL_MAP = {

    "bonafide": 0,

    "spoof": 1,

}

# ======================================================
# PATHS
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (

    PROJECT_ROOT

    / "datasets"

    / "melspec"

    / "anti_spoof"

)

OUTPUT_DIR = (

    PROJECT_ROOT

    / "outputs"

    / "stage1"

)

OUTPUT_DIR.mkdir(

    parents=True,

    exist_ok=True,

)


# ======================================================
# MAIN
# ======================================================

def main():

    # ==================================================
    # DEVICE
    # ==================================================

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )

    if torch.cuda.is_available():

        torch.backends.cudnn.benchmark = True

    print()

    print("=" * 70)

    print(f"DEVICE: {device}")

    print("=" * 70)

    # ==================================================
    # DATASET
    # ==================================================

    train_dataset = SpeechDataset(

        DATASET_DIR,

        "train",

        "binary_label",

        LABEL_MAP,

    )
    debug_dataset(

        train_dataset

    )
    
    print()

    print(

        train_dataset.df.groupby(

            "binary_label"

        ).head(3)[

            [

                "filename",

                "binary_label",

            ]

        ]

    )

    val_dataset = SpeechDataset(

        DATASET_DIR,

        "val",

        "binary_label",

        LABEL_MAP,

    )

    test_dataset = SpeechDataset(

        DATASET_DIR,

        "test",

        "binary_label",

        LABEL_MAP,

    )

    # ==================================================
    # DATALOADER
    # ==================================================

    loader_kwargs = {

        "batch_size": BATCH_SIZE,

        "num_workers": NUM_WORKERS,

        "pin_memory": torch.cuda.is_available(),

    }

    if torch.cuda.is_available():

        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(

        train_dataset,

        shuffle=True,

        **loader_kwargs,

    )

    x, y = next(

        iter(

            train_loader

        )

    )

    print()

    print("=" * 70)

    print("FIRST BATCH")

    print("=" * 70)

    print()

    print(

        x.shape

    )

    print(

        y[:20]

    )

    val_loader = DataLoader(

        val_dataset,

        shuffle=False,

        **loader_kwargs,

    )

    test_loader = DataLoader(

        test_dataset,

        shuffle=False,

        **loader_kwargs,

    )

    # ==================================================
    # MODEL
    # ==================================================

    model = CNN(

        num_classes=2

    ).to(device)
    
    print()

    print("="*70)

    print("MODEL")

    print("="*70)

    print(model)

    print()

    print(

        "Trainable params:",

        sum(

            p.numel()

            for p in model.parameters()

            if p.requires_grad

        )

    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE,

    )

    scheduler = ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=2,

    )

    scaler = torch.amp.GradScaler(

        device="cuda",

        enabled=torch.cuda.is_available(),

    )

    # ==================================================
    # TRAIN
    # ==================================================

    history = []

    best_f1 = 0

    early_stop_counter = 0

    print()

    print("=" * 70)

    print("START TRAINING")

    print("=" * 70)

    for epoch in range(EPOCHS):

        model.train()

        train_correct = 0

        train_total = 0

        train_loss = 0

        first_weight_before = (

            model.features[0]

            .weight[0,0,0,0]

            .item()

        )
        
        for x, y in tqdm(

            train_loader,

            desc=f"Epoch {epoch+1}",

        ):

            x = x.to(device)

            y = y.to(device)

            optimizer.zero_grad(

                set_to_none=True

            )

            with torch.amp.autocast(

                device_type=device.type,

                enabled=torch.cuda.is_available(),

            ):

                outputs = model(x)

                loss = criterion(

                    outputs,

                    y,

                )

            scaler.scale(

                loss

            ).backward()

            scaler.step(

                optimizer

            )

            scaler.update()

            train_loss += loss.item()

            preds = outputs.argmax(

                dim=1

            )

            train_correct += (

                preds == y

            ).sum().item()

            train_total += y.size(0)

        train_acc = (

            train_correct

            / train_total

        )
        first_weight_after = (

            model.features[0]

            .weight[0,0,0,0]

            .item()

        )

        print()

        print(

            "Weight:",

            first_weight_before,

            "->",

            first_weight_after,

        )

        # ==============================================
        # VALIDATION
        # ==============================================

        model.eval()

        y_true = []

        y_pred = []

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(device)

                outputs = model(x)

                preds = outputs.argmax(

                    dim=1

                )

                y_true.extend(

                    y.numpy()

                )

                y_pred.extend(

                    preds.cpu().numpy()

                )

        val_f1 = f1_score(

            y_true,

            y_pred,

            average="macro",

        )

        scheduler.step(

            val_f1

        )

        current_lr = optimizer.param_groups[0]["lr"]

        history.append({

            "epoch": epoch + 1,

            "train_acc": train_acc,

            "val_f1": val_f1,

            "lr": current_lr,

        })

        print()

        print(

            f"Epoch {epoch+1}/{EPOCHS}"

        )

        print(

            f"Train Acc: {train_acc:.4f}"

        )

        print(

            f"Val F1: {val_f1:.4f}"

        )

        print(

            f"LR: {current_lr:.6f}"

        )

        # ==============================================
        # SAVE BEST MODEL
        # ==============================================

        if val_f1 > best_f1:

            best_f1 = val_f1

            early_stop_counter = 0

            torch.save(

                model.state_dict(),

                OUTPUT_DIR

                / "best_model.pth",

            )

            print(

                "Saved best model"

            )

        else:

            early_stop_counter += 1

        # ==============================================
        # EARLY STOPPING
        # ==============================================

        if early_stop_counter >= PATIENCE:

            print()

            print(

                "Early stopping"

            )

            break

    # ==================================================
    # SAVE HISTORY
    # ==================================================

    history_df = pd.DataFrame(

        history

    )

    history_df.to_csv(

        OUTPUT_DIR

        / "history.csv",

        index=False,

    )

    # ==================================================
    # LEARNING CURVE
    # ==================================================

    plt.figure(

        figsize=(8, 5)

    )

    plt.plot(

        history_df["epoch"],

        history_df["train_acc"],

    )

    plt.plot(

        history_df["epoch"],

        history_df["val_f1"],

    )

    plt.legend(

        [

            "train_acc",

            "val_f1",

        ]

    )

    plt.xlabel("Epoch")

    plt.ylabel("Score")

    plt.grid()

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR

        / "learning_curve.png"

    )

    plt.close()

    # ==================================================
    # TEST
    # ==================================================

    print()

    print("=" * 70)

    print("TESTING")

    print("=" * 70)

    model.load_state_dict(

        torch.load(

            OUTPUT_DIR

            / "best_model.pth",

            map_location=device,

            weights_only=True,

        )

    )

    model.eval()

    y_true = []

    y_pred = []

    with torch.no_grad():

        for x, y in test_loader:

            x = x.to(device)

            outputs = model(x)

            preds = outputs.argmax(

                dim=1

            )

            y_true.extend(

                y.numpy()

            )

            y_pred.extend(

                preds.cpu().numpy()

            )

    metrics = evaluate(

        y_true,

        y_pred,

        OUTPUT_DIR,

    )

    print()

    print("=" * 70)

    print("FINAL RESULT")

    print("=" * 70)

    for key, value in metrics.items():

        print(

            f"{key}: {value:.4f}"

        )

    print()

    print("Saved to:")

    print(OUTPUT_DIR)


if __name__ == "__main__":

    main()