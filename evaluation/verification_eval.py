from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import torch

import torch.nn.functional as F

from torch.utils.data import DataLoader

from sklearn.metrics import (

    roc_curve,

    roc_auc_score,

    accuracy_score,

)

from src.verification_pairs import (

    generate_verification_pairs,

)


# =====================================================
# COSINE SIMILARITY
# =====================================================

def cosine_similarity(

    emb1,

    emb2,

):

    return F.cosine_similarity(

        emb1,

        emb2,

        dim=0,

    ).item()


# =====================================================
# EER
# =====================================================

def compute_eer(

    y_true,

    y_score,

):

    fpr, tpr, thresholds = roc_curve(

        y_true,

        y_score,

    )

    fnr = 1 - tpr

    idx = np.nanargmin(

        np.abs(

            fpr - fnr

        )

    )

    eer = (

        fpr[idx]

        + fnr[idx]

    ) / 2

    threshold = thresholds[idx]

    return (

        eer,

        threshold,

        fpr,

        tpr,

    )


# =====================================================
# EXTRACT EMBEDDINGS
# =====================================================

def extract_embeddings(

    model,

    dataset,

    device,

    batch_size=64,

    num_workers=4,

):

    loader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=torch.cuda.is_available(),

    )

    embeddings = {}

    model.eval()

    with torch.no_grad():

        for batch in loader:

            mel = batch["mel"].to(

                device

            )

            filenames = batch[

                "filename"

            ]

            _, emb = model(

                mel,

                return_embedding=True,

            )

            emb = emb.cpu()

            for i, filename in enumerate(

                filenames

            ):

                embeddings[

                    filename

                ] = emb[i]

    return embeddings


# =====================================================
# MAIN EVALUATION
# =====================================================

def verification_evaluate(

    model,

    dataset,

    labels_csv,

    device,

    output_dir,

):

    output_dir = Path(

        output_dir

    )

    output_dir.mkdir(

        parents=True,

        exist_ok=True,

    )

    embeddings = extract_embeddings(

        model,

        dataset,

        device,

    )

    pairs = generate_verification_pairs(

        labels_csv,

        split="test",

    )

    y_true = []

    y_score = []

    pair_results = []

    for _, row in pairs.iterrows():

        emb1 = embeddings[

            row["file1"]

        ]

        emb2 = embeddings[

            row["file2"]

        ]

        score = cosine_similarity(

            emb1,

            emb2,

        )

        label = row["pair_label"]

        y_true.append(

            label

        )

        y_score.append(

            score

        )

        pair_results.append({

            "file1": row["file1"],

            "file2": row["file2"],

            "pair_label": label,

            "similarity_score": score,

        })

    auc = roc_auc_score(

        y_true,

        y_score,

    )

    (

        eer,

        threshold,

        fpr,

        tpr,

    ) = compute_eer(

        y_true,

        y_score,

    )

    y_pred = (

        np.array(

            y_score

        )

        >= threshold

    ).astype(int)

    verification_acc = accuracy_score(

        y_true,

        y_pred,

    )

    # ==========================================
    # METRICS
    # ==========================================

    metrics = {

        "auc": auc,

        "eer": eer,

        "threshold": threshold,

        "verification_accuracy": verification_acc,

        "num_pairs": len(

            y_true

        ),

    }

    pd.DataFrame(

        [metrics]

    ).to_csv(

        output_dir

        / "verification_metrics.csv",

        index=False,

    )

    # ==========================================
    # SAVE PAIRS
    # ==========================================

    pd.DataFrame(

        pair_results

    ).to_csv(

        output_dir

        / "pair_scores.csv",

        index=False,

    )

    # ==========================================
    # REPORT
    # ==========================================

    with open(

        output_dir

        / "verification_report.txt",

        "w",

        encoding="utf-8",

    ) as f:

        for k, v in metrics.items():

            f.write(

                f"{k}: {v}\n"

            )

    # ==========================================
    # ROC CURVE
    # ==========================================

    plt.figure(

        figsize=(6,6)

    )

    plt.plot(

        fpr,

        tpr,

    )

    plt.plot(

        [0,1],

        [0,1],

        "--",

    )

    plt.xlabel(

        "False Positive Rate"

    )

    plt.ylabel(

        "True Positive Rate"

    )

    plt.title(

        f"ROC Curve (AUC={auc:.4f})"

    )

    plt.grid()

    plt.tight_layout()

    plt.savefig(

        output_dir

        / "verification_roc_curve.png",

        dpi=300,

    )

    plt.close()

    # ==========================================
    # SCORE DISTRIBUTION
    # ==========================================

    y_true = np.array(

        y_true

    )

    y_score = np.array(

        y_score

    )

    positive_scores = y_score[

        y_true == 1

    ]

    negative_scores = y_score[

        y_true == 0

    ]

    plt.figure(

        figsize=(8,5)

    )

    plt.hist(

        positive_scores,

        bins=50,

        alpha=0.6,

    )

    plt.hist(

        negative_scores,

        bins=50,

        alpha=0.6,

    )

    plt.axvline(

        threshold,

        linestyle="--",

    )

    plt.legend([

        "positive",

        "negative",

        "eer threshold",

    ])

    plt.xlabel(

        "Cosine Similarity"

    )

    plt.ylabel(

        "Frequency"

    )

    plt.title(

        "Score Distribution"

    )

    plt.grid()

    plt.tight_layout()

    plt.savefig(

        output_dir

        / "score_distribution.png",

        dpi=300,

    )

    plt.close()

    return metrics