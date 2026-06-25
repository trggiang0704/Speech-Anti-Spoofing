from pathlib import Path

import numpy as np
import pandas as pd

import torch

from torch.utils.data import Dataset


class SpeechDataset(Dataset):

    def __init__(

        self,

        dataset_dir,

        split,

        label_column,

        label_map,

        specaugment=None,

    ):

        self.dataset_dir = Path(

            dataset_dir

        )

        self.specaugment = specaugment

        self.label_column = label_column

        self.label_map = label_map

        self.split = split

        self.df = pd.read_csv(

            self.dataset_dir

            / "labels.csv"

        )

        self.df = self.df[

            self.df["split"]

            == split

        ]

        self.df = self.df.reset_index(

            drop=True

        )

    def __len__(self):

        return len(

            self.df

        )

    def __getitem__(

        self,

        idx,

    ):

        row = self.df.iloc[idx]

        mel_path = (

            self.dataset_dir

            / row["filename"]

        ).with_suffix(

            ".npy"

        )

        mel = np.load(

            mel_path

        )

        mel = torch.tensor(

            mel,

            dtype=torch.float32,

        )

        mel = mel.unsqueeze(0)

        # ==========================
        # SpecAugment
        # train only
        # ==========================

        if self.specaugment is not None:

            mel = self.specaugment(

                mel.unsqueeze(0)

            )

            mel = mel.squeeze(0)

        label = self.label_map[

            row[self.label_column]

        ]

        return {

            "mel": mel,

            "label": torch.tensor(

                label,

                dtype=torch.long,

            ),

            "speaker": row["speaker"],

            "filename": row["filename"],

        }


# ======================================================
# DEBUG
# ======================================================

def debug_dataset(

    dataset,

):

    print()

    print("=" * 70)

    print("DATASET DEBUG")

    print("=" * 70)

    print()

    print(

        f"Total: {len(dataset)}"

    )

    print()

    sample = dataset[0]

    print(

        sample["mel"].shape

    )

    print(

        sample["label"]

    )

    print(

        sample["speaker"]

    )