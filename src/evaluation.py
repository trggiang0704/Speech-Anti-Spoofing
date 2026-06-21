from pathlib import Path

import pandas as pd

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    confusion_matrix,

    classification_report,

)

import matplotlib.pyplot as plt


def evaluate(

    y_true,

    y_pred,

    output_dir,

):

    output_dir = Path(

        output_dir

    )

    output_dir.mkdir(

        parents=True,

        exist_ok=True,

    )

    metrics = {

        "accuracy": accuracy_score(

            y_true,

            y_pred,

        ),

        "precision_macro": precision_score(

            y_true,

            y_pred,

            average="macro",

        ),

        "recall_macro": recall_score(

            y_true,

            y_pred,

            average="macro",

        ),

        "f1_macro": f1_score(

            y_true,

            y_pred,

            average="macro",

        ),

    }

    pd.DataFrame(

        [metrics]

    ).to_csv(

        output_dir

        / "metrics.csv",

        index=False,

    )

    report = classification_report(

        y_true,

        y_pred,

        digits=4,

    )

    with open(

        output_dir

        / "classification_report.txt",

        "w",

        encoding="utf-8",

    ) as f:

        f.write(

            report

        )

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        cm,
        interpolation="nearest",
    )

    plt.colorbar()

    classes = [
        "bonafide",
        "spoof",
    ]

    plt.xticks(
        range(len(classes)),
        classes,
    )

    plt.yticks(
        range(len(classes)),
        classes,
    )

    # Hiện số

    threshold = cm.max() / 2

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            plt.text(

                j,

                i,

                str(cm[i, j]),

                ha="center",

                va="center",

                color=(
                    "white"

                    if cm[i, j] > threshold

                    else "black"

                ),

                fontsize=14,

                fontweight="bold",

            )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "True"
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(

        output_dir

        / "confusion_matrix.png"

    )

    plt.close()

    return metrics