# Dataset

Dataset files are not included in this repository.

## Source

VSASV Dataset:

https://huggingface.co/datasets/hustep-lab/VSASV-Dataset

## Generate Dataset

```bash
python src/create_dataset.py
```

### Generated Dataset

- Total samples: 50,000
- Speaker-disjoint split
- Balanced across 3 classes:

  - `bonafide`
  - `adversarial_attack`
  - `voice_conversion`

### Split Ratio

- Train: 80%
- Validation: 10%
- Test: 10%

### Output Structure

```text
datasets/
└── raw/
    └── VSASV_PAPER_50000/
        ├── train/
        ├── val/
        ├── test/
        └── labels.csv
```

> Dataset files are excluded from GitHub due to their size.