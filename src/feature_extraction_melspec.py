from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from tqdm import tqdm

# ======================================================
# CONFIG
# ======================================================

SR = 16000

N_FFT = 1024

HOP_LENGTH = 256

N_MELS = 80

DURATION = 4

MAX_LEN = int(np.ceil((SR * DURATION) / HOP_LENGTH))

DATASET_NAME = "VSASV_PAPER_50000"

# ======================================================
# PATHS
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "datasets"

INPUT_DIR = (

    DATASET_ROOT

    / "processed"

    / DATASET_NAME

)

OUTPUT_DIR = (

    DATASET_ROOT

    / "logmelspec"

    / DATASET_NAME

)

LABELS_PATH = (

    INPUT_DIR

    / "labels.csv"

)

# ======================================================
# CHECK
# ======================================================

if not INPUT_DIR.exists():

    raise FileNotFoundError(

        f"Cannot find {INPUT_DIR}"

    )

if not LABELS_PATH.exists():

    raise FileNotFoundError(

        f"Cannot find {LABELS_PATH}"

    )

# ======================================================
# LOG-MEL
# ======================================================

def create_logmel(audio_path):

    audio, _ = librosa.load(

        audio_path,

        sr=SR,

        mono=True,

    )

    mel = librosa.feature.melspectrogram(

        y=audio,

        sr=SR,

        n_fft=N_FFT,

        hop_length=HOP_LENGTH,

        n_mels=N_MELS,

        power=2.0,

    )

    mel = librosa.power_to_db(

        mel,

        ref=np.max,

    )

    # z-score normalization

    mel = (

        mel

        - mel.mean()

    ) / (

        mel.std()

        + 1e-8

    )

    width = mel.shape[1]

    if width < MAX_LEN:

        pad = MAX_LEN - width

        mel = np.pad(

            mel,

            (

                (0, 0),

                (0, pad),

            ),

            mode="constant",

        )

    else:

        mel = mel[:, :MAX_LEN]

    return mel.astype(

        np.float32

    )

# ======================================================
# MAIN
# ======================================================

def main():

    print()

    print("=" * 70)

    print("CREATE LOG-MEL SPECTROGRAM")

    print("=" * 70)

    print()

    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True,

    )

    labels = pd.read_csv(

        LABELS_PATH

    )

    print(

        f"Total samples: {len(labels)}"

    )

    print()

    for _, row in tqdm(

        labels.iterrows(),

        total=len(labels),

    ):

        audio_path = (

            INPUT_DIR

            / row["filename"]

        )

        save_path = (

            OUTPUT_DIR

            / row["filename"]

        ).with_suffix(

            ".npy"

        )

        save_path.parent.mkdir(

            parents=True,

            exist_ok=True,

        )

        if save_path.exists():

            continue

        mel = create_logmel(

            audio_path

        )

        np.save(

            save_path,

            mel,

        )

    labels.to_csv(

        OUTPUT_DIR

        / "labels.csv",

        index=False,

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