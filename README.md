# Vietnamese Speech Anti-Spoofing & Speaker Verification

A deep learning framework for **Vietnamese Speech Anti-Spoofing** and **Speaker Verification** using Log-Mel Spectrogram and CNN.

The project adopts a two-stage architecture:

- **Stage 1:** Multi-class Speech Anti-Spoofing
- **Stage 2:** Speaker Verification using shared embeddings

---

## Project Overview

The system is designed to detect spoofed Vietnamese speech while simultaneously evaluating speaker identity.

Unlike traditional binary anti-spoofing systems, this project formulates spoof detection as a **multi-class classification** problem consisting of:

- Bonafide
- Voice Conversion
- Adversarial Attack

The learned embeddings are then reused for speaker verification without retraining the backbone.

---

## Pipeline

```
Raw Audio
      │
      ▼
Audio Preprocessing
      │
      ▼
Log-Mel Spectrogram
      │
      ▼
SpecAugment (training only)
      │
      ▼
CNN Backbone
      │
      ├──────────────► Stage 1
      │                Multi-class Anti-Spoofing
      │
      └──────────────► Stage 2
                       Speaker Verification
```

![Pipeline dự án](outputs\image.png)

---

## Dataset

Source:

- VSASV Dataset

Dataset after processing:

- 50,000 audio samples
- WAV format
- 16 kHz
- 4-second duration
- Speaker-independent train/validation/test split

Classes:

- Bonafide
- Voice Conversion
- Adversarial Attack

Dataset split:

| Split | Samples |
|--------|---------|
| Train | 40,000 |
| Validation | 5,000 |
| Test | 5,000 |

---

## Audio Preprocessing

The preprocessing pipeline includes:

- Resampling to 16 kHz
- Mono conversion
- DC offset removal
- Silence trimming
- Amplitude normalization
- Fixed-length padding/truncation (4 seconds)

---

## Feature Extraction

Each audio sample is converted into an **80 × 250 Log-Mel Spectrogram**.

Configuration:

- Sampling Rate: 16 kHz
- n_fft: 1024
- Hop Length: 256
- Mel Filters: 80

---

## Data Augmentation

SpecAugment is applied during training using:

- Frequency Masking
- Time Masking

This improves model robustness and reduces overfitting.

---

## Model Architecture

The model consists of:

- CNN Backbone
- Global Average Pooling
- 128-dimensional Shared Embedding
- Classification Head

The shared embedding is reused for speaker verification using cosine similarity.

---

## Project Structure

```text
project/
│
├── datasets/
│   ├── raw/
│   ├── processed/
│   └── logmelspec/
│
├── src/
│   ├── create_dataset.py
│   ├── preprocess.py
│   ├── extract_logmel.py
│   ├── train_stage1.py
│   ├── evaluate_stage2.py
│   └── ...
│
├── models/
├── outputs/
├── checkpoints/
└── README.md
```

---

## Training

Generate dataset

```bash
python src/create_dataset.py
```

Preprocess audio

```bash
python src/preprocess.py
```

Extract Log-Mel Spectrogram

```bash
python src/extract_logmel.py
```

Train Stage 1

```bash
python src/train_stage1.py
```

Evaluate Speaker Verification

```bash
python src/evaluate_stage2.py
```

---

## Experimental Results

### Stage 1 – Speech Anti-Spoofing

| Metric | Value |
|---------|------:|
| Accuracy | 99.90% |
| Macro F1 | 99.90% |
| CM-EER | 0.06% |

![Loss Curve](outputs\stage1_cnn\loss_curve.png)

![Validation Metrics Curve](outputs\stage1_cnn\metric_curve.png)

![Confusion Matrix](outputs\stage1_cnn\confusion_matrix.png)

---

### Stage 2 – Speaker Verification

| Metric | Value |
|---------|------:|
| Verification Accuracy | 82.74% |
| AUC | 88.25% |
| EER | 17.29% |

![ROC Curve](outputs\stage2_verification\verification_roc_curve.png)

![Score Distribution](outputs\stage2_verification\score_distribution.png)

---

## Features

- Multi-class speech anti-spoofing
- Speaker-independent dataset split
- Shared embedding architecture
- Log-Mel Spectrogram representation
- SpecAugment for data augmentation
- Speaker verification using cosine similarity

---

## Future Work

- Train with larger Vietnamese speech datasets
- Explore ECAPA-TDNN and ResNet backbones
- Evaluate against additional spoofing attacks
- End-to-end joint optimization for anti-spoofing and speaker verification

---

## License

This project is intended for academic and research purposes.