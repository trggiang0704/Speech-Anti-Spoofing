import os
import glob
import random

import pandas as pd
import soundfile as sf

from tqdm import tqdm
from pathlib import Path


# =======================================================
# CONFIG
# =======================================================

TOTAL_SAMPLES = 50000

TRAIN_RATIO = 0.8

VAL_RATIO = 0.1

TEST_RATIO = 0.1

BONAFIDE_RATIO = 0.5

ADV_RATIO = 0.25

VC_RATIO = 0.25

SEED = 42

random.seed(SEED)

VALID_TYPES = [

    "bonafide",

    "adversarial_attack",

    "voice_conversion",

]

# =======================================================
# PATHS
# =======================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "datasets"

HF_CACHE_DIR = Path(r"F:\hf_cache")

snapshot_dir = next(

    (

        HF_CACHE_DIR

        / "hub"

        / "datasets--hustep-lab--VSASV-Dataset"

        / "snapshots"

    ).iterdir()

)

PARQUET_DIR = snapshot_dir / "data"

RAW_DIR = DATASET_ROOT / "raw"

OUTPUT_DIR = RAW_DIR / f"VSASV_{TOTAL_SAMPLES}"

# =======================================================
# TARGET
# =======================================================

train_total = int(

    TOTAL_SAMPLES * TRAIN_RATIO

)

val_total = int(

    TOTAL_SAMPLES * VAL_RATIO

)

test_total = (

    TOTAL_SAMPLES

    - train_total

    - val_total

)

TARGET = {

    "train": {

        "bonafide":

            int(

                train_total

                * BONAFIDE_RATIO

            ),

        "adversarial_attack":

            int(

                train_total

                * ADV_RATIO

            ),

        "voice_conversion":

            train_total

            - int(

                train_total

                * BONAFIDE_RATIO

            )

            - int(

                train_total

                * ADV_RATIO

            ),

    },

    "val": {

        "bonafide":

            int(

                val_total

                * BONAFIDE_RATIO

            ),

        "adversarial_attack":

            int(

                val_total

                * ADV_RATIO

            ),

        "voice_conversion":

            val_total

            - int(

                val_total

                * BONAFIDE_RATIO

            )

            - int(

                val_total

                * ADV_RATIO

            ),

    },

    "test": {

        "bonafide":

            int(

                test_total

                * BONAFIDE_RATIO

            ),

        "adversarial_attack":

            int(

                test_total

                * ADV_RATIO

            ),

        "voice_conversion":

            test_total

            - int(

                test_total

                * BONAFIDE_RATIO

            )

            - int(

                test_total

                * ADV_RATIO

            ),

    },

}

# =======================================================
# PARQUET FILES
# =======================================================

parquet_files = glob.glob(

    str(

        PARQUET_DIR

        / "*.parquet"

    )

)

random.shuffle(

    parquet_files

)

# =======================================================
# PASS 1
# COLLECT SPEAKERS
# =======================================================

print()

print("=" * 70)

print("COLLECT SPEAKERS")

print("=" * 70)

speaker_set = set()

for pq in tqdm(

    parquet_files

):

    df = pd.read_parquet(

        pq,

        columns=[

            "label",

            "utt_type",

        ],

    )

    df = df[

        df["utt_type"]

        .isin(

            VALID_TYPES

        )

    ]

    speaker_set.update(

        df["label"]

        .unique()

    )

    del df

# =======================================================
# SPLIT SPEAKERS
# =======================================================

speakers = list(

    speaker_set

)

random.shuffle(

    speakers

)

n = len(

    speakers

)

train_speakers = set(

    speakers[

        :int(

            n

            * TRAIN_RATIO

        )

    ]

)

val_speakers = set(

    speakers[

        int(

            n

            * TRAIN_RATIO

        ):

        int(

            n

            * (

                TRAIN_RATIO

                + VAL_RATIO

            )

        )

    ]

)

test_speakers = set(

    speakers[

        int(

            n

            * (

                TRAIN_RATIO

                + VAL_RATIO

            )

        ):

    ]

)

print()

print("=" * 70)

print("SPEAKER SPLIT")

print("=" * 70)

print(

    "train:",

    len(

        train_speakers

    )

)

print(

    "val:",

    len(

        val_speakers

    )

)

print(

    "test:",

    len(

        test_speakers

    )

)

# =======================================================
# CREATE FOLDERS
# =======================================================

for split in [

    "train",

    "val",

    "test",

]:

    (

        OUTPUT_DIR

        / split

        / "bonafide"

    ).mkdir(

        parents=True,

        exist_ok=True,

    )

    (

        OUTPUT_DIR

        / split

        / "spoof"

    ).mkdir(

        parents=True,

        exist_ok=True,

    )

# =======================================================
# COUNTERS
# =======================================================

count = {

    split: {

        cls: 0

        for cls

        in VALID_TYPES

    }

    for split in [

        "train",

        "val",

        "test",

    ]

}

labels = []

# =======================================================
# PASS 2
# CREATE DATASET
# =======================================================

print()

print("=" * 70)

print(

    f"CREATE VSASV_{TOTAL_SAMPLES}"

)

print("=" * 70)

finished = False

for pq in tqdm(

    parquet_files

):

    df = pd.read_parquet(

        pq

    )

    df = df.sample(

        frac=1,

        random_state=SEED,

    )

    for _, row in df.iterrows():

        original_type = row["utt_type"]

        if original_type not in VALID_TYPES:

            continue

        speaker = row["label"]

        if speaker in train_speakers:

            split = "train"

        elif speaker in val_speakers:

            split = "val"

        else:

            split = "test"

        if (

            count[split]

            [original_type]

            >=

            TARGET[split]

            [original_type]

        ):

            continue

        binary_label = (

            "bonafide"

            if original_type

            == "bonafide"

            else "spoof"

        )

        index = (

            count[split]

            [original_type]

            + 1

        )

        if original_type == "bonafide":

            filename = (

                f"bf_{index:05d}.wav"

            )

        elif original_type == "adversarial_attack":

            filename = (

                f"adv_{index:05d}.wav"

            )

        else:

            filename = (

                f"vc_{index:05d}.wav"

            )

        relative_path = (

            Path(split)

            / binary_label

            / filename

        )

        save_path = (

            OUTPUT_DIR

            / relative_path

        )

        audio = row["audio"]["array"]

        sr = row["audio"]["sampling_rate"]

        sf.write(

            save_path,

            audio,

            sr,

        )

        labels.append({

            "filename":

                relative_path

                .as_posix(),

            "speaker":

                speaker,

            "binary_label":

                binary_label,

            "original_type":

                original_type,

            "split":

                split,

        })

        count[split][

            original_type

        ] += 1

        finished = all(

            count[s][cls]

            >=

            TARGET[s][cls]

            for s in TARGET

            for cls in TARGET[s]

        )

        if finished:

            break

    del df

    if finished:

        break

# =======================================================
# SAVE LABELS
# =======================================================

labels_df = pd.DataFrame(

    labels

)

labels_df = labels_df.sample(

    frac=1,

    random_state=SEED,

).reset_index(

    drop=True

)

labels_df.to_csv(

    OUTPUT_DIR

    / "labels.csv",

    index=False,

    encoding="utf-8-sig",

)

# =======================================================
# SUMMARY
# =======================================================

print()

print("=" * 70)

print("DONE")

print("=" * 70)

print()

for split in count:

    print(split)

    print(

        count[split]

    )

    print()

print(

    "Total:",

    len(labels_df),

)

print()

print("Saved to:")

print(

    OUTPUT_DIR

)