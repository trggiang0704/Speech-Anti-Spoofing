from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import wandb

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from tqdm import tqdm

from src.dataset import SpeechDataset
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

DATASET_NAME = "VSASV_50000"

MODEL_NAME = "CNN"

LABEL_MAP = {

    "adversarial_attack": 0,

    "voice_conversion": 1,

}

TARGET_LABELS = [

    "adversarial_attack",

    "voice_conversion",

]


# ======================================================
# PATHS
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (

    PROJECT_ROOT

    / "datasets"

    / "melspec"

    / DATASET_NAME

)

OUTPUT_DIR = (

    PROJECT_ROOT

    / "outputs"

    / f"stage2_{MODEL_NAME.lower()}"

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
    # SEED
    # ==================================================

    SEED = 42

    torch.manual_seed(SEED)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(SEED)

    # ==================================================
    # W&B
    # ==================================================

    wandb.init(

        project="speech-anti-spoofing",

        name=f"stage2_{MODEL_NAME.lower()}",

        config={

            "dataset": DATASET_NAME,

            "model": MODEL_NAME,

            "batch_size": BATCH_SIZE,

            "epochs": EPOCHS,

            "learning_rate": LEARNING_RATE,

        },

    )

    # ==================================================
    # DATASET
    # ==================================================

    train_dataset = SpeechDataset(
        DATASET_DIR,
        "train",
        "original_type",
        LABEL_MAP,
        allowed_labels= TARGET_LABELS
    )

    val_dataset = SpeechDataset(

        DATASET_DIR,

        "val",

        "original_type",

        LABEL_MAP,

        allowed_labels= TARGET_LABELS
    )

    test_dataset = SpeechDataset(

        DATASET_DIR,

        "test",

        "original_type",

        LABEL_MAP,

        allowed_labels= TARGET_LABELS
    )

    print()

    print("=" * 70)

    print("DATASET")

    print("=" * 70)

    print(f"Train: {len(train_dataset)}")

    print(f"Val:   {len(val_dataset)}")

    print(f"Test:  {len(test_dataset)}")

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

    print("=" * 70)

    print("MODEL")

    print("=" * 70)

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

        # ==============================================
        # TRAIN
        # ==============================================

        model.train()

        train_loss = 0

        train_correct = 0

        train_total = 0

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

        train_loss /= len(

            train_loader

        )

        train_acc = (

            train_correct

            / train_total

        )

        # ==============================================
        # VALIDATION
        # ==============================================

        model.eval()

        val_loss = 0

        y_true = []

        y_pred = []

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(device)

                y = y.to(device)

                outputs = model(x)

                loss = criterion(

                    outputs,

                    y,

                )

                val_loss += loss.item()

                preds = outputs.argmax(

                    dim=1

                )

                y_true.extend(

                    y.cpu().numpy()

                )

                y_pred.extend(

                    preds.cpu().numpy()

                )

        val_loss /= len(

            val_loader

        )

        val_acc = accuracy_score(

            y_true,

            y_pred,

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

            "train_loss": train_loss,

            "train_acc": train_acc,

            "val_loss": val_loss,

            "val_acc": val_acc,

            "val_f1": val_f1,

            "lr": current_lr,

        })

        print()

        print(f"Epoch {epoch+1}/{EPOCHS}")

        print(f"Train Loss: {train_loss:.4f}")

        print(f"Train Acc : {train_acc:.4f}")

        print(f"Val Loss  : {val_loss:.4f}")

        print(f"Val Acc   : {val_acc:.4f}")

        print(f"Val F1    : {val_f1:.4f}")

        print(f"LR        : {current_lr:.6f}")

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

            print("Saved best model")

        else:

            early_stop_counter += 1
            
        wandb.log({

            "epoch": epoch + 1,

            "train_loss": train_loss,

            "train_acc": train_acc,

            "val_loss": val_loss,

            "val_acc": val_acc,

            "val_f1": val_f1,

            "best_val_f1": best_f1,

            "lr": current_lr,

        })

        # ==============================================
        # EARLY STOPPING
        # ==============================================

        if early_stop_counter >= PATIENCE:

            print()

            print("Early stopping")

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

        history_df["train_loss"],

    )

    plt.plot(

        history_df["epoch"],

        history_df["val_loss"],

    )

    plt.plot(

        history_df["epoch"],

        history_df["val_f1"],

    )

    plt.legend(

        [

            "train_loss",

            "val_loss",

            "val_f1",

        ]

    )

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

        class_names=TARGET_LABELS,

    )

    wandb.log({

        "test_accuracy": metrics["accuracy"],

        "test_precision_macro": metrics["precision_macro"],

        "test_recall_macro": metrics["recall_macro"],

        "test_f1_macro": metrics["f1_macro"],

    }) 
    
    wandb.finish()

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