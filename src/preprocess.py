from pathlib import Path
import shutil

import librosa
import numpy as np
import soundfile as sf

from tqdm import tqdm


# =======================================================
# CONFIG
# =======================================================

TARGET_SR = 16000

TARGET_DURATION = 4

TARGET_LENGTH = TARGET_SR * TARGET_DURATION

DATASET_NAME = "VSASV_50000"

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
# FUNCTIONS
# =======================================================

def preprocess_audio(audio):

    # mono

    if audio.ndim > 1:

        audio = np.mean(

            audio,

            axis=1,

        )

    # normalize

    max_amp = np.max(

        np.abs(audio)

    )

    if max_amp > 0:

        audio = audio / max_amp

    # padding

    if len(audio) < TARGET_LENGTH:

        pad = TARGET_LENGTH - len(audio)

        audio = np.pad(

            audio,

            (0, pad),

            mode="constant",

        )

    # crop

    elif len(audio) > TARGET_LENGTH:

        audio = audio[:TARGET_LENGTH]

    return audio


# =======================================================
# MAIN
# =======================================================

print()

print("=" * 70)

print("PREPROCESS DATASET")

print("=" * 70)

print(DATASET_NAME)

# =======================================================
# CREATE OUTPUT FOLDER
# =======================================================

OUTPUT_DIR.mkdir(

    parents=True,

    exist_ok=True,

)

# copy labels

shutil.copy2(

    LABELS_PATH,

    OUTPUT_DIR / "labels.csv",

)

# =======================================================
# LOAD FILES
# =======================================================

wav_files = list(

    INPUT_DIR.rglob("*.wav")

)

print()

print(f"Total wav files: {len(wav_files)}")

print()

# =======================================================
# PREPROCESS
# =======================================================

for wav_path in tqdm(wav_files):

    relative_path = wav_path.relative_to(

        INPUT_DIR

    )

    save_path = (

        OUTPUT_DIR

        / relative_path

    )

    save_path.parent.mkdir(

        parents=True,

        exist_ok=True,

    )

    # skip if existed

    if save_path.exists():

        continue

    audio, _ = librosa.load(

        wav_path,

        sr=TARGET_SR,

        mono=True,

    )

    audio = preprocess_audio(

        audio

    )

    sf.write(

        save_path,

        audio,

        TARGET_SR,

    )

print()

print("=" * 70)

print("DONE")

print("=" * 70)

print()

print("Saved to:")

print(OUTPUT_DIR)