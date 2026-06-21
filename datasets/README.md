# Datasets

Dataset files are not included in this repository.

## Source

VSASV Dataset:

https://huggingface.co/datasets/hustep-lab/VSASV-Dataset

## Generate Dataset

```bash
python src/create_dataset.py
```

Output:

```text
raw/
└── VSASV_50000/
```

## Preprocess Audio

```bash
python src/preprocess.py
```

Output:

```text
processed/
└── VSASV_50000/
```

## Create Mel Spectrograms

```bash
python src/create_melspec.py
```

Output:

```text
melspec/
└── VSASV_50000/
```

## Data Split

- Train: 80%
- Validation: 10%
- Test: 10%

## Labels

- Stage 1: `bonafide`, `spoof`
- Stage 2: `adversarial_attack`, `voice_conversion`

> Dataset files are excluded from GitHub due to their size.