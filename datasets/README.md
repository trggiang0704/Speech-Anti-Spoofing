# Dataset

The dataset used in this project is not included in this repository due to its large size.

## Source

Original dataset: https://huggingface.co/datasets/hustep-lab/VSASV-Dataset

## Reproduction

The processed dataset can be generated using:

```bash
python src/create_dataset.py
```

The script:

* Reads the original VSASV dataset from `.parquet` files.
* Creates a binary anti-spoofing dataset (`bonafide` vs `spoof`).
* Splits the data into train (80%), validation (10%), and test (10%).
* Saves audio files as `.wav`.
* Generates a `labels.csv` file.

The generated dataset structure is:

```text
VSASV_20000/
├── train/
│   ├── bonafide/
│   └── spoof/
├── val/
│   ├── bonafide/
│   └── spoof/
├── test/
│   ├── bonafide/
│   └── spoof/
└── labels.csv
```

> Note: Large dataset files are intentionally excluded from GitHub.
