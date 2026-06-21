import os
import glob
import random
import pandas as pd
import soundfile as sf
from tqdm import tqdm

# =======================================================
# CONFIG
# =======================================================

PARQUET_DIR = r"F:\DoAn\datasets\hf_cache\hub\datasets--hustep-lab--VSASV-Dataset\snapshots\92de668616780ed4f3d7f58a82262f82b56b4557\data"

TOTAL_SAMPLES = 20000

OUTPUT_DIR = rf"F:\DoAn\datasets\VSASV_{TOTAL_SAMPLES}"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

BONAFIDE_RATIO = 0.5
ADV_RATIO = 0.25
VC_RATIO = 0.25

train_total = int(TOTAL_SAMPLES * TRAIN_RATIO)
val_total = int(TOTAL_SAMPLES * VAL_RATIO)
test_total = TOTAL_SAMPLES - train_total - val_total

TARGET = {

    "train": {

        "bonafide": int(train_total * BONAFIDE_RATIO),

        "adversarial_attack": int(train_total * ADV_RATIO),

        "voice_conversion":
            train_total
            - int(train_total * BONAFIDE_RATIO)
            - int(train_total * ADV_RATIO),

    },

    "val": {

        "bonafide": int(val_total * BONAFIDE_RATIO),

        "adversarial_attack": int(val_total * ADV_RATIO),

        "voice_conversion":
            val_total
            - int(val_total * BONAFIDE_RATIO)
            - int(val_total * ADV_RATIO),

    },

    "test": {

        "bonafide": int(test_total * BONAFIDE_RATIO),

        "adversarial_attack": int(test_total * ADV_RATIO),

        "voice_conversion":
            test_total
            - int(test_total * BONAFIDE_RATIO)
            - int(test_total * ADV_RATIO),

    },

}

random.seed(42)
# =======================================================

parquet_files = glob.glob(os.path.join(PARQUET_DIR, "*.parquet"))
random.shuffle(parquet_files)

# tạo thư mục

for split in ["train", "val", "test"]:

    os.makedirs(
        os.path.join(OUTPUT_DIR, split, "bonafide"),
        exist_ok=True,
    )

    os.makedirs(
        os.path.join(OUTPUT_DIR, split, "spoof"),
        exist_ok=True,
    )

count = {

    "train": {
        "bonafide": 0,
        "adversarial_attack": 0,
        "voice_conversion": 0,
    },

    "val": {
        "bonafide": 0,
        "adversarial_attack": 0,
        "voice_conversion": 0,
    },

    "test": {
        "bonafide": 0,
        "adversarial_attack": 0,
        "voice_conversion": 0,
    },

}

labels = []


def choose_split(original_type):

    for split in ["train", "val", "test"]:

        if count[split][original_type] < TARGET[split][original_type]:
            return split

    return None


print("=" * 70)
print(f"CREATE VSASV_{TOTAL_SAMPLES}")
print("=" * 70)

for pq in tqdm(parquet_files):

    df = pd.read_parquet(pq)

    df = df.sample(frac=1, random_state=42)

    for _, row in df.iterrows():

        original_type = row["utt_type"]

        if original_type not in [
            "bonafide",
            "adversarial_attack",
            "voice_conversion",
        ]:
            continue

        split = choose_split(original_type)

        if split is None:
            continue

        binary_label = (
            "bonafide"
            if original_type == "bonafide"
            else "spoof"
        )

        index = count[split][original_type] + 1

        if original_type == "bonafide":
            filename = f"bf_{index:05d}.wav"
        elif original_type == "adversarial_attack":
            filename = f"adv_{index:05d}.wav"
        else:
            filename = f"vc_{index:05d}.wav"

        relative_path = os.path.join(
            split,
            binary_label,
            filename,
        )

        save_path = os.path.join(
            OUTPUT_DIR,
            relative_path,
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
                relative_path.replace("\\", "/"),

            "speaker":
                row["label"],

            "binary_label":
                binary_label,

            "original_type":
                original_type,

            "split":
                split,

        })

        count[split][original_type] += 1

    finished = True

    for split in TARGET:
        for cls in TARGET[split]:

            if count[split][cls] < TARGET[split][cls]:
                finished = False

    if finished:
        break

labels_df = pd.DataFrame(labels)

labels_df = labels_df.sample(
    frac=1,
    random_state=42,
).reset_index(drop=True)

labels_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "labels.csv",
    ),

    index=False,

    encoding="utf-8-sig",

)

print()
print("=" * 70)
print("DONE")
print("=" * 70)
print()

print(labels_df.head())

print()

print("STATISTICS")

print()

for split in count:

    print(split)

    print(count[split])

    print()

print("Saved to:")

print(OUTPUT_DIR)

print()

print("Target:")

for split in TARGET:
    print(split, TARGET[split])