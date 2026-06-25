from pathlib import Path

import random

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from tqdm import tqdm

import wandb

from models.cnn import CNN

from src.dataset import SpeechDataset

from src.specaugment import SpecAugment

from src.evaluation.classification_eval import (
    evaluate,
    compute_eer,
)

# ======================================================
# CONFIG
# ======================================================

SEED = 42

BATCH_SIZE = 64

EPOCHS = 20

PATIENCE = 7

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 4

DATASET_NAME = "VSASV_PAPER_50000"

MODEL_NAME = "CNN"

LABEL_MAP = {

    "bonafide": 0,

    "adversarial_attack": 1,

    "voice_conversion": 2,

}

CLASS_NAMES = [

    "bonafide",

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

    / "logmelspec"

    / DATASET_NAME

)

OUTPUT_DIR = (

    PROJECT_ROOT

    / "outputs"

    / f"stage1_{MODEL_NAME.lower()}"

)

OUTPUT_DIR.mkdir(

    parents=True,

    exist_ok=True,

)

# ======================================================
# REPRODUCIBILITY
# ======================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ======================================================
# CM-EER BINARY MAPPING
# ======================================================

def multiclass_to_binary(labels):

    labels = np.array(labels)

    binary = np.where(

        labels == 0,

        0,

        1,

    )

    return binary


# ======================================================
# VALIDATION
# ======================================================

def run_validation(

    model,

    loader,

    criterion,

    device,

):

    model.eval()

    y_true = []

    y_pred = []

    y_score = []

    total_loss = 0

    with torch.no_grad():

        for batch in loader:

            x = batch["mel"].to(device)

            y = batch["label"].to(device)

            outputs = model(x)

            loss = criterion(

                outputs,

                y,

            )

            total_loss += loss.item()

            probs = F.softmax(

                outputs,

                dim=1,

            )

            preds = outputs.argmax(

                dim=1

            )

            y_true.extend(

                y.cpu().numpy()

            )

            y_pred.extend(

                preds.cpu().numpy()

            )

            spoof_score = (

                probs[:,1]

                +

                probs[:,2]

            )

            y_score.extend(

                spoof_score

                .cpu()

                .numpy()

            )

    loss = total_loss / len(loader)

    acc = accuracy_score(

        y_true,

        y_pred,

    )

    precision = precision_score(

        y_true,

        y_pred,

        average="macro",

        zero_division=0,

    )

    recall = recall_score(

        y_true,

        y_pred,

        average="macro",

        zero_division=0,

    )

    f1 = f1_score(

        y_true,

        y_pred,

        average="macro",

        zero_division=0,

    )

    binary_true = multiclass_to_binary(

        y_true

    )

    cm_eer, _ = compute_eer(

        binary_true,

        y_score,

    )

    return {

        "loss": loss,

        "accuracy": acc,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "cm_eer": cm_eer,

    }


# ======================================================
# MAIN
# ======================================================

def main():

    set_seed(SEED)

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )

    print()

    print("=" * 70)

    print(f"DEVICE: {device}")

    print("=" * 70)

    # ==================================================
    # WANDB
    # ==================================================

    wandb.init(

        project="speech-anti-spoofing",

        name=f"{MODEL_NAME}_{DATASET_NAME}",

        config={

            "seed": SEED,

            "batch_size": BATCH_SIZE,

            "epochs": EPOCHS,

            "learning_rate": LEARNING_RATE,

            "weight_decay": WEIGHT_DECAY,

            "dataset": DATASET_NAME,

            "model": MODEL_NAME,

        },

    )

    # ==================================================
    # DATASET
    # ==================================================

    train_dataset = SpeechDataset(

        DATASET_DIR,

        "train",

        "label",

        LABEL_MAP,

    )

    val_dataset = SpeechDataset(

        DATASET_DIR,

        "val",

        "label",

        LABEL_MAP,

    )

    test_dataset = SpeechDataset(

        DATASET_DIR,

        "test",

        "label",

        LABEL_MAP,

    )

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

        num_classes=3,

        embedding_dim=128,

    ).to(device)

    specaugment = SpecAugment()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

    )

    scheduler = ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=2,

    )

    # ==================================================
    # TRAIN
    # ==================================================

    history = []

    best_score = -1

    patience_counter = 0

    for epoch in range(EPOCHS):

        model.train()
        
        specaugment.train()

        train_loss = 0

        train_correct = 0

        train_total = 0

        for batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}",
        ):

            x = batch["mel"].to(device)

            y = batch["label"].to(device)

            x = specaugment(x)

            optimizer.zero_grad(

                set_to_none=True

            )

            outputs = model(x)

            loss = criterion(

                outputs,

                y,

            )

            loss.backward()

            optimizer.step()

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

        val_metrics = run_validation(

            model,

            val_loader,

            criterion,

            device,

        )

        scheduler.step(

            val_metrics["f1"]

        )

        selection_score = (

            0.7

            * val_metrics["f1"]

            +

            0.3

            * (1 - val_metrics["cm_eer"])

        )

        history.append({

            "epoch": epoch + 1,

            "train_loss": train_loss,

            "train_acc": train_acc,

            "val_loss": val_metrics["loss"],

            "val_acc": val_metrics["accuracy"],

            "val_precision": val_metrics["precision"],

            "val_recall": val_metrics["recall"],

            "val_f1": val_metrics["f1"],

            "val_cm_eer": val_metrics["cm_eer"],

            "selection_score": selection_score,

            "lr": optimizer.param_groups[0]["lr"],

        })

        wandb.log(

            history[-1]

        )

        print()

        print(

            f"Epoch {epoch+1}/{EPOCHS}"

        )

        print(

            f"Train Acc: {train_acc:.4f}"

        )
        
        print(

            f"Train Loss: {train_loss:.4f}"

        )

        print(

            f"Val Loss: {val_metrics['loss']:.4f}"

        )

        print(

            f"Val Acc: {val_metrics['accuracy']:.4f}"

        )

        print(

            f"Val Precision: {val_metrics['precision']:.4f}"

        )

        print(

            f"Val Recall: {val_metrics['recall']:.4f}"

        )

        print(

            f"Val F1: {val_metrics['f1']:.4f}"

        )

        print(

            f"Val CM-EER: {val_metrics['cm_eer']:.4f}"

        )

        print(

            f"Selection Score: {selection_score:.4f}"

        )

        if selection_score > best_score:

            best_score = selection_score

            patience_counter = 0

            torch.save(

                model.state_dict(),

                OUTPUT_DIR

                / "best_model.pth",

            )
            print(

                f"Saved best model | Selection Score = {selection_score:.4f}"

            )

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

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
    # Loss curve
    plt.figure(

        figsize=(8,5)

    )

    plt.plot(

        history_df["epoch"],

        history_df["train_loss"],

    )

    plt.plot(

        history_df["epoch"],

        history_df["val_loss"],

    )

    plt.legend([

        "train_loss",

        "val_loss",

    ])

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.grid()

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR

        / "loss_curve.png"

    )

    plt.close()
    
    # Metric curve
    plt.figure(

        figsize=(8,5)

    )

    plt.plot(

        history_df["epoch"],

        history_df["val_f1"],

    )

    plt.plot(

        history_df["epoch"],

        history_df["val_cm_eer"],

    )

    plt.plot(

        history_df["epoch"],

        history_df["selection_score"],

    )

    plt.legend([

        "val_f1",

        "val_cm_eer",

        "selection_score",

    ])

    plt.xlabel("Epoch")

    plt.ylabel("Metric")

    plt.grid()

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR

        / "metric_curve.png"

    )

    plt.close()
    
    

    # ==================================================
    # TEST
    # ==================================================

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

    y_score = []

    with torch.no_grad():

        for batch in test_loader:

            x = batch["mel"].to(device)

            y = batch["label"]

            outputs = model(x)

            probs = F.softmax(

                outputs,

                dim=1,

            )

            preds = outputs.argmax(

                dim=1

            )

            y_true.extend(

                y.numpy()

            )

            y_pred.extend(

                preds.cpu().numpy()

            )

            spoof_score = (

                probs[:,1]

                +

                probs[:,2]

            )

            y_score.extend(

                spoof_score

                .cpu()

                .numpy()

            )

    binary_true = multiclass_to_binary(

        y_true

    )

    metrics = evaluate(

        y_true=y_true,

        y_pred=y_pred,

        output_dir=OUTPUT_DIR,

        class_names=CLASS_NAMES,

        y_true_binary=binary_true,

        y_spoof_score=y_score,

    )

    print()

    print("=" * 70)

    print("FINAL RESULT")

    print("=" * 70)

    for k, v in metrics.items():

        print(

            f"{k}: {v:.4f}"

        )

    wandb.finish()


if __name__ == "__main__":

    main()