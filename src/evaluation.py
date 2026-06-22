from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate(
    y_true,
    y_pred,
    output_dir,
    class_names=None,
):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==================================================
    # METRICS
    # ==================================================

    metrics = {

        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

    }

    pd.DataFrame(
        [metrics]
    ).to_csv(

        output_dir
        / "metrics.csv",

        index=False,

    )

    # ==================================================
    # CLASSIFICATION REPORT
    # ==================================================

    report = classification_report(

        y_true,

        y_pred,

        digits=4,

        zero_division=0,

    )

    with open(

        output_dir
        / "classification_report.txt",

        "w",

        encoding="utf-8",

    ) as f:

        f.write(report)

    # ==================================================
    # CONFUSION MATRIX
    # ==================================================

    cm = confusion_matrix(

        y_true,

        y_pred,

    )

    if class_names is None:

        class_names = [

            str(i)

            for i in range(

                len(cm)

            )

        ]

    plt.figure(

        figsize=(6, 6)

    )

    plt.imshow(

        cm,

        interpolation="nearest",

    )

    plt.colorbar()

    plt.xticks(

        range(len(class_names)),

        class_names,

    )

    plt.yticks(

        range(len(class_names)),

        class_names,

    )

    threshold = cm.max() / 2

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            plt.text(

                j,

                i,

                str(cm[i, j]),

                ha="center",

                va="center",

                fontsize=14,

                fontweight="bold",

                color=(

                    "white"

                    if cm[i, j] > threshold

                    else "black"

                ),

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
        / "confusion_matrix.png",

        dpi=300,

    )

    plt.close()

    return metrics