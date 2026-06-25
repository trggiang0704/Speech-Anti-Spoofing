import os
import glob
import random
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from pathlib import Path

# =======================================================
# CONFIG (FULLY REPRODUCIBLE)
# =======================================================

SEED = 42
random.seed(SEED)

TOTAL_SAMPLES = 50000

SPLITS = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1
}

CLASSES = [
    "bonafide",
    "adversarial_attack",
    "voice_conversion"
]

# exact per split
NUM_CLASSES = len(CLASSES)

# =======================================================
# PATHS
# =======================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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

OUTPUT_DIR = PROJECT_ROOT / "datasets" / "raw" / "VSASV_PAPER_50000"

# =======================================================
# EXACT TARGET COMPUTATION
# =======================================================

def compute_targets():
    targets = {}

    per_split_total = {
        s: int(TOTAL_SAMPLES * r)
        for s, r in SPLITS.items()
    }

    for split, size in per_split_total.items():
        per_class = size // NUM_CLASSES

        targets[split] = {
            cls: per_class for cls in CLASSES
        }

        # fix remainder (to guarantee exact total)
        remainder = size - per_class * NUM_CLASSES

        for i in range(remainder):
            targets[split][CLASSES[i % NUM_CLASSES]] += 1

    return targets


TARGET = compute_targets()

# =======================================================
# LOAD DATA
# =======================================================

parquet_files = glob.glob(str(PARQUET_DIR / "*.parquet"))
random.shuffle(parquet_files)

# =======================================================
# COLLECT SPEAKERS
# =======================================================

print("=" * 70)
print("COLLECT SPEAKERS")
print("=" * 70)

speaker_set = set()

for pq in tqdm(parquet_files):
    df = pd.read_parquet(pq, columns=["label", "utt_type"])
    df = df[df["utt_type"].isin(CLASSES)]
    speaker_set.update(df["label"].unique())

speakers = list(speaker_set)
random.shuffle(speakers)

# speaker split (STRICT)
n = len(speakers)

train_speakers = set(speakers[: int(n * 0.8)])
val_speakers = set(speakers[int(n * 0.8): int(n * 0.9)])
test_speakers = set(speakers[int(n * 0.9):])

# =======================================================
# FOLDERS
# =======================================================

for split in SPLITS:
    for cls in CLASSES:
        (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

# =======================================================
# COUNTERS
# =======================================================

count = {
    split: {cls: 0 for cls in CLASSES}
    for split in SPLITS
}

labels = []

# =======================================================
# HELPER
# =======================================================

def get_split(speaker):
    if speaker in train_speakers:
        return "train"
    elif speaker in val_speakers:
        return "val"
    else:
        return "test"

def is_full(split):
    return all(
        count[split][cls] >= TARGET[split][cls]
        for cls in CLASSES
    )

# =======================================================
# BUILD DATASET (STRICT CONTROL)
# =======================================================

print("=" * 70)
print("BUILDING PAPER-GRADE DATASET")
print("=" * 70)

finished = False

for pq in tqdm(parquet_files):

    df = pd.read_parquet(pq)
    df = df.sample(frac=1, random_state=SEED)

    for _, row in df.iterrows():

        cls = row["utt_type"]
        if cls not in CLASSES:
            continue

        speaker = row["label"]
        split = get_split(speaker)

        # STOP IF SPLIT IS FULL
        if is_full(split):
            continue

        # STOP IF CLASS IS FULL IN THAT SPLIT
        if count[split][cls] >= TARGET[split][cls]:
            continue

        index = count[split][cls] + 1

        filename = f"{cls[:3]}_{index:05d}.wav"

        rel_path = Path(split) / cls / filename
        save_path = OUTPUT_DIR / rel_path

        audio = row["audio"]["array"]
        sr = row["audio"]["sampling_rate"]

        sf.write(save_path, audio, sr)

        labels.append({
            "filename": rel_path.as_posix(),
            "speaker": speaker,
            "label": cls,
            "split": split
        })

        count[split][cls] += 1

        # global stop condition (STRICT)
        if all(
            count[s][c] == TARGET[s][c]
            for s in SPLITS
            for c in CLASSES
        ):
            finished = True
            break

    if finished:
        break

# =======================================================
# SAVE LABELS
# =======================================================

df = pd.DataFrame(labels)
df = df.sample(frac=1, random_state=SEED)

df.to_csv(
    OUTPUT_DIR / "labels.csv",
    index=False,
    encoding="utf-8-sig"
)

# =======================================================
# SUMMARY CHECK
# =======================================================

print("=" * 70)
print("FINAL DISTRIBUTION (STRICT PAPER-GRADE)")
print("=" * 70)

for split in SPLITS:
    print(f"\n{split}")
    print(count[split])
    print("TOTAL:", sum(count[split].values()))

print("\nOVERALL:", len(df))
print("SAVED TO:", OUTPUT_DIR)