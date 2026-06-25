from pathlib import Path

import shutil

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from tqdm import tqdm


# =======================================================
# CONFIG
# =======================================================

TARGET_SR = 16000

TARGET_DURATION = 4

TARGET_LENGTH = TARGET_SR * TARGET_DURATION

DATASET_NAME = "VSASV_PAPER_50000"

# =======================================================
# PATHS
# =======================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "datasets"

RAW_ROOT = DATASET_ROOT / "raw"

PROCESSED_ROOT = DATASET_ROOT / "processed"

INPUT_DIR = RAW_ROOT / DATASET_NAME

OUTPUT_DIR = PROCESSED_ROOT / DATASET_NAME

LABELS_PATH = INPUT_DIR / "labels.csv"

OUTPUT_LABELS = OUTPUT_DIR / "labels.csv"

# =======================================================
# CHECK
# =======================================================

if not INPUT_DIR.exists():

    raise FileNotFoundError(
        f"Cannot find {INPUT_DIR}"
    )

if not LABELS_PATH.exists():

    raise FileNotFoundError(
        f"Cannot find {LABELS_PATH}"
    )

# =======================================================
# AUDIO LOADER
# =======================================================

def load_audio(file_path):

    audio, _ = librosa.load(

        file_path,

        sr=TARGET_SR,

        mono=True,

    )

    return audio


# =======================================================
# PREPROCESS AUDIO
# =======================================================

def preprocess_audio(audio):

    if audio is None:

        return None

    if len(audio) == 0:

        return None

    if not np.all(np.isfinite(audio)):

        return None

    # ==========================================
    # remove dc offset
    # ==========================================

    audio = audio - np.mean(audio)

    # ==========================================
    # normalize amplitude
    # ==========================================

    max_amp = np.max(

        np.abs(audio)

    )

    if max_amp > 0:

        audio = audio / (

            max_amp + 1e-8

        )

    # ==========================================
    # fixed length = 4 seconds
    # ==========================================

    if len(audio) < TARGET_LENGTH:

        pad = TARGET_LENGTH - len(audio)

        audio = np.pad(

            audio,

            (0, pad),

            mode="constant",

        )

    else:

        audio = audio[:TARGET_LENGTH]

    return audio.astype(

        np.float32

    )


# =======================================================
# PREPROCESS DATASET
# =======================================================

def preprocess_dataset():

    print()

    print("=" * 70)

    print("STEP 2 - AUDIO PREPROCESSING")

    print("=" * 70)

    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True,

    )

    labels = pd.read_csv(

        LABELS_PATH

    )

    print()

    print(

        f"Total samples: {len(labels)}"

    )

    error_count = 0

    for _, row in tqdm(

        labels.iterrows(),

        total=len(labels),

    ):

        try:

            relative_path = Path(

                row["filename"]

            )

            input_path = (

                INPUT_DIR

                / relative_path

            )

            output_path = (

                OUTPUT_DIR

                / relative_path

            )

            output_path.parent.mkdir(

                parents=True,

                exist_ok=True,

            )

            if output_path.exists():

                continue

            audio = load_audio(

                input_path

            )

            audio = preprocess_audio(

                audio

            )

            if audio is None:

                error_count += 1

                continue

            sf.write(

                output_path,

                audio,

                TARGET_SR,

            )

        except Exception as e:

            error_count += 1

            print()

            print(

                f"[ERROR] {input_path}"

            )

            print(e)

            continue

    # ==================================================
    # COPY LABELS
    # ==================================================

    shutil.copy2(

        LABELS_PATH,

        OUTPUT_LABELS,

    )

    print()

    print("=" * 70)

    print("DONE")

    print("=" * 70)

    print()

    print(

        f"Errors: {error_count}"

    )

    print()

    print(

        f"Saved to: {OUTPUT_DIR}"

    )


# =======================================================
# MAIN
# =======================================================

if __name__ == "__main__":

    preprocess_dataset()