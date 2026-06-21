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

    ):

        self.dataset_dir = Path(

            dataset_dir

        )

        self.label_column = label_column

        self.label_map = label_map

        self.df = pd.read_csv(

            self.dataset_dir

            / "labels.csv"

        )

        self.df = self.df[

            self.df["split"] == split

        ].reset_index(

            drop=True

        )

    def __len__(self):

        return len(self.df)

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

        label = self.label_map[

            row[self.label_column]

        ]

        return (

            mel,

            torch.tensor(

                label,

                dtype=torch.long,

            ),

        )


# ======================================================
# DEBUG FUNCTION
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

        f"Total samples: {len(dataset)}"

    )

    print()

    for i in range(5):

        x, y = dataset[i]

        print(

            f"sample={i}"

        )

        print(

            f"shape={x.shape}"

        )

        print(

            f"label={y.item()}"

        )

        print(

            f"min={x.min():.4f}"

        )

        print(

            f"max={x.max():.4f}"

        )

        print()