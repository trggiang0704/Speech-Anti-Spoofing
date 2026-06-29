from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)

# ==========================================================
# CONFIG
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STAGE2_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "stage2_verification"
)

PAIR_SCORE_FILE = (
    STAGE2_DIR
    / "pair_scores.csv"
)

METRIC_FILE = (
    STAGE2_DIR
    / "verification_metrics.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "error_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TOP_ERROR = 10

# ==========================================================
# LOAD DATA
# ==========================================================

def load_data():

    if not PAIR_SCORE_FILE.exists():
        raise FileNotFoundError(
            PAIR_SCORE_FILE
        )

    if not METRIC_FILE.exists():
        raise FileNotFoundError(
            METRIC_FILE
        )

    pair_df = pd.read_csv(
        PAIR_SCORE_FILE
    )

    metric_df = pd.read_csv(
        METRIC_FILE
    )

    threshold = float(
        metric_df["threshold"].iloc[0]
    )

    return pair_df, threshold


# ==========================================================
# PREDICTION
# ==========================================================

def predict(pair_df, threshold):

    pair_df = pair_df.copy()

    pair_df["prediction"] = (
        pair_df["similarity_score"]
        >= threshold
    ).astype(int)

    pair_df["correct"] = (
        pair_df["prediction"]
        ==
        pair_df["pair_label"]
    )

    return pair_df


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

def save_confusion_matrix(pair_df):

    y_true = pair_df["pair_label"]

    y_pred = pair_df["prediction"]

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Different",
            "Same",
        ],
    )

    plt.figure(figsize=(5,5))

    disp.plot()

    plt.title(
        "Speaker Verification Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "confusion_matrix.png",
        dpi=300,
    )

    plt.close()


# ==========================================================
# ROC
# ==========================================================

def save_roc(pair_df):

    fpr, tpr, _ = roc_curve(
        pair_df["pair_label"],
        pair_df["similarity_score"],
    )

    plt.figure(figsize=(6,6))

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
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
        "Verification ROC Curve"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "roc_curve.png",
        dpi=300,
    )

    plt.close()


# ==========================================================
# SCORE DISTRIBUTION
# ==========================================================

def save_score_distribution(
    pair_df,
    threshold,
):

    positive = pair_df[
        pair_df["pair_label"] == 1
    ]

    negative = pair_df[
        pair_df["pair_label"] == 0
    ]

    plt.figure(figsize=(8,5))

    plt.hist(
        positive["similarity_score"],
        bins=40,
        alpha=0.6,
        label="Positive Pair",
    )

    plt.hist(
        negative["similarity_score"],
        bins=40,
        alpha=0.6,
        label="Negative Pair",
    )

    plt.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label="Threshold",
    )

    plt.xlabel(
        "Cosine Similarity"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Similarity Distribution"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "score_distribution.png",
        dpi=300,
    )

    plt.close()
    
    # ==========================================================
# FALSE ACCEPTANCE (FP)
# ==========================================================

def analyze_false_acceptance(
    pair_df,
    threshold,
):

    fp = pair_df[
        (pair_df["pair_label"] == 0)
        &
        (pair_df["prediction"] == 1)
    ].copy()

    if len(fp) == 0:

        print("No False Acceptance.")

        return

    # khoảng cách tới threshold
    fp["distance"] = (
        fp["similarity_score"]
        - threshold
    )

    fp = fp.sort_values(
        "distance",
        ascending=False,
    )

    top_fp = fp.head(
        TOP_ERROR
    )

    top_fp.to_csv(
        OUTPUT_DIR
        / "false_acceptance.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print("TOP FALSE ACCEPTANCE")
    print("=" * 70)

    print(
        top_fp[
            [
                "file1",
                "file2",
                "similarity_score",
                "distance",
            ]
        ]
    )


# ==========================================================
# FALSE REJECTION (FN)
# ==========================================================

def analyze_false_rejection(
    pair_df,
    threshold,
):

    fn = pair_df[
        (pair_df["pair_label"] == 1)
        &
        (pair_df["prediction"] == 0)
    ].copy()

    if len(fn) == 0:

        print("No False Rejection.")

        return

    fn["distance"] = (
        threshold
        -
        fn["similarity_score"]
    )

    fn = fn.sort_values(
        "distance",
        ascending=False,
    )

    top_fn = fn.head(
        TOP_ERROR
    )

    top_fn.to_csv(
        OUTPUT_DIR
        / "false_rejection.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print("TOP FALSE REJECTION")
    print("=" * 70)

    print(
        top_fn[
            [
                "file1",
                "file2",
                "similarity_score",
                "distance",
            ]
        ]
    )


# ==========================================================
# HARDEST POSITIVE PAIRS
# ==========================================================

def analyze_hard_positive(
    pair_df,
):

    positive = pair_df[
        pair_df["pair_label"] == 1
    ].copy()

    positive = positive.sort_values(
        "similarity_score",
        ascending=True,
    )

    hardest = positive.head(
        TOP_ERROR
    )

    hardest.to_csv(
        OUTPUT_DIR
        / "hard_positive_pairs.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print("HARDEST POSITIVE PAIRS")
    print("=" * 70)

    print(
        hardest[
            [
                "file1",
                "file2",
                "similarity_score",
            ]
        ]
    )


# ==========================================================
# HARDEST NEGATIVE PAIRS
# ==========================================================

def analyze_hard_negative(
    pair_df,
):

    negative = pair_df[
        pair_df["pair_label"] == 0
    ].copy()

    negative = negative.sort_values(
        "similarity_score",
        ascending=False,
    )

    hardest = negative.head(
        TOP_ERROR
    )

    hardest.to_csv(
        OUTPUT_DIR
        / "hard_negative_pairs.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print("HARDEST NEGATIVE PAIRS")
    print("=" * 70)

    print(
        hardest[
            [
                "file1",
                "file2",
                "similarity_score",
            ]
        ]
    )
    
# ==========================================================
# ERROR BY AUDIO TYPE
# ==========================================================

def analyze_audio_type(pair_df):

    def get_type(path):

        if "bonafide" in path:
            return "bonafide"

        if "adversarial_attack" in path:
            return "adversarial_attack"

        return "voice_conversion"

    pair_df = pair_df.copy()

    pair_df["type1"] = pair_df["file1"].apply(get_type)
    pair_df["type2"] = pair_df["file2"].apply(get_type)

    error = pair_df[
        pair_df["correct"] == False
    ]

    rows = []

    for t in [
        "bonafide",
        "adversarial_attack",
        "voice_conversion",
    ]:

        total = (
            (pair_df["type1"] == t)
            |
            (pair_df["type2"] == t)
        ).sum()

        wrong = (
            (error["type1"] == t)
            |
            (error["type2"] == t)
        ).sum()

        rows.append({

            "audio_type": t,

            "total_pairs": total,

            "error_pairs": wrong,

            "error_rate": wrong / total

            if total > 0 else 0,

        })

    df = pd.DataFrame(rows)

    df.to_csv(

        OUTPUT_DIR /

        "audio_type_statistics.csv",

        index=False,

    )

    print()

    print("="*70)

    print("ERROR BY AUDIO TYPE")

    print("="*70)

    print(df)
    
# ==========================================================
# TOP ERROR CASES
# ==========================================================

def analyze_error_cases(

    pair_df,

    threshold,

):

    errors = pair_df[

        pair_df["correct"] == False

    ].copy()

    errors["distance"] = np.abs(

        errors["similarity_score"]

        -

        threshold

    )

    errors = errors.sort_values(

        "distance",

        ascending=False,

    )

    errors = errors.head(

        TOP_ERROR

    )

    rows = []

    for _, row in errors.iterrows():

        if row["pair_label"] == 0:

            reason = (

                "Different speakers nhưng similarity rất cao."

            )

        else:

            reason = (

                "Same speaker nhưng similarity quá thấp."

            )

        rows.append({

            "file1": row["file1"],

            "file2": row["file2"],

            "pair_label": row["pair_label"],

            "prediction": row["prediction"],

            "similarity": row["similarity_score"],

            "reason": reason,

        })

    pd.DataFrame(

        rows

    ).to_csv(

        OUTPUT_DIR

        / "top10_error_cases.csv",

        index=False,

    )

    print()

    print("="*70)

    print("TOP ERROR CASES")

    print("="*70)

    print(pd.DataFrame(rows))
    
# ==========================================================
# SUMMARY REPORT
# ==========================================================

def generate_report(
    pair_df,
    threshold,
):

    total_pairs = len(pair_df)

    correct = int(pair_df["correct"].sum())

    incorrect = total_pairs - correct

    fp = len(
        pair_df[
            (pair_df["pair_label"] == 0)
            &
            (pair_df["prediction"] == 1)
        ]
    )

    fn = len(
        pair_df[
            (pair_df["pair_label"] == 1)
            &
            (pair_df["prediction"] == 0)
        ]
    )

    report = []

    report.append("=" * 70)
    report.append("SPEAKER VERIFICATION ERROR ANALYSIS")
    report.append("=" * 70)
    report.append("")

    report.append(f"Total verification pairs : {total_pairs}")
    report.append(f"Correct predictions      : {correct}")
    report.append(f"Incorrect predictions    : {incorrect}")
    report.append(f"Threshold (EER)          : {threshold:.6f}")
    report.append("")

    report.append("False Acceptance (FP)")
    report.append(f"    {fp}")

    report.append("")

    report.append("False Rejection (FN)")
    report.append(f"    {fn}")

    report.append("")

    report.append("=" * 70)
    report.append("MAIN ERROR ANALYSIS")
    report.append("=" * 70)

    report.append("")
    report.append("1. False Acceptance")
    report.append(
        "- Different speakers nhưng embedding quá giống nhau."
    )
    report.append(
        "- Có thể do đặc trưng ngữ âm hoặc đặc điểm phổ của hai speaker gần nhau."
    )
    report.append(
        "- Cosine Similarity vượt quá ngưỡng nên hệ thống nhận nhầm cùng người nói."
    )

    report.append("")
    report.append("2. False Rejection")
    report.append(
        "- Hai audio cùng speaker nhưng embedding bị tách xa."
    )
    report.append(
        "- Có thể do chất lượng ghi âm, tốc độ nói, ngữ điệu hoặc nhiễu môi trường."
    )

    report.append("")
    report.append("3. Hard Positive Pairs")
    report.append(
        "- Positive pair có similarity thấp nhất."
    )
    report.append(
        "- Đây là các trường hợp khó nhất đối với mô hình."
    )

    report.append("")
    report.append("4. Hard Negative Pairs")
    report.append(
        "- Negative pair có similarity cao nhất."
    )
    report.append(
        "- Đây là nguyên nhân chính gây False Acceptance."
    )

    report.append("")
    report.append("=" * 70)
    report.append("POSSIBLE IMPROVEMENTS")
    report.append("=" * 70)

    report.append("")
    report.append(
        "- Sử dụng Additive Margin Softmax / ArcFace để tăng khoảng cách embedding."
    )
    report.append(
        "- Hard Negative Mining trong quá trình huấn luyện."
    )
    report.append(
        "- Bổ sung dữ liệu speaker và nhiều điều kiện ghi âm hơn."
    )
    report.append(
        "- Huấn luyện bằng metric learning (Triplet Loss hoặc Contrastive Loss)."
    )
    report.append(
        "- Kết hợp backbone mạnh hơn như ECAPA-TDNN hoặc ResNet."
    )

    with open(
        OUTPUT_DIR / "analysis_report.txt",
        "w",
        encoding="utf-8",
    ) as f:

        for line in report:
            f.write(line + "\n")


# ==========================================================
# MAIN
# ==========================================================

def main():

    print()
    print("=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    pair_df, threshold = load_data()

    pair_df = predict(
        pair_df,
        threshold,
    )

    save_confusion_matrix(
        pair_df,
    )

    save_roc(
        pair_df,
    )

    save_score_distribution(
        pair_df,
        threshold,
    )

    analyze_false_acceptance(
        pair_df,
        threshold,
    )

    analyze_false_rejection(
        pair_df,
        threshold,
    )

    analyze_hard_positive(
        pair_df,
    )

    analyze_hard_negative(
        pair_df,
    )
    
    analyze_audio_type(
        pair_df
    )
    
    analyze_error_cases(

        pair_df,

        threshold,

    )

    generate_report(
        pair_df,
        threshold,
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print()
    print("Saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":

    main()
    
    