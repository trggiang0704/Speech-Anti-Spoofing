from itertools import combinations

import random

import pandas as pd


def generate_verification_pairs(

    labels_csv,

    split="test",

    positive_pairs_per_speaker=20,

    negative_pairs_per_speaker=20,

    seed=42,

):

    random.seed(seed)

    df = pd.read_csv(

        labels_csv

    )

    df = df[

        df["split"] == split

    ].reset_index(

        drop=True

    )

    speakers = df[

        "speaker"

    ].unique()

    positive_pairs = []

    negative_pairs = []

    # ==========================================
    # POSITIVE
    # ==========================================

    for speaker in speakers:

        speaker_df = df[

            df["speaker"]

            == speaker

        ]

        indices = speaker_df.index.tolist()

        if len(indices) < 2:

            continue

        all_pairs = list(

            combinations(

                indices,

                2,

            )

        )

        random.shuffle(

            all_pairs

        )

        all_pairs = all_pairs[

            :positive_pairs_per_speaker

        ]

        for i, j in all_pairs:

            positive_pairs.append({

                "file1": df.loc[i, "filename"],

                "file2": df.loc[j, "filename"],

                "pair_label": 1,

            })

    # ==========================================
    # NEGATIVE
    # ==========================================

    for speaker in speakers:

        anchor_df = df[

            df["speaker"]

            == speaker

        ]

        other_df = df[

            df["speaker"]

            != speaker

        ]

        for _, anchor in anchor_df.iterrows():

            sampled = other_df.sample(

                n=min(

                    negative_pairs_per_speaker,

                    len(other_df),

                ),

                replace=False,

                random_state=seed,

            )

            for _, other in sampled.iterrows():

                negative_pairs.append({

                    "file1": anchor["filename"],

                    "file2": other["filename"],

                    "pair_label": 0,

                })

    pair_df = pd.DataFrame(

        positive_pairs

        + negative_pairs

    )

    pair_df = pair_df.sample(

        frac=1,

        random_state=seed,

    ).reset_index(

        drop=True

    )

    return pair_df