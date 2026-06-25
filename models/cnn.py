import torch

import torch.nn as nn

import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(

        self,

        num_classes=3,

        embedding_dim=128,

    ):

        super().__init__()

        # =====================
        # CNN BACKBONE
        # =====================

        self.backbone = nn.Sequential(

            nn.Conv2d(

                1,

                32,

                kernel_size=3,

                padding=1,

            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(

                32,

                64,

                kernel_size=3,

                padding=1,

            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(

                64,

                128,

                kernel_size=3,

                padding=1,

            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.Dropout(0.3),

        )

        # =====================
        # GLOBAL POOLING
        # =====================

        self.global_pool = (

            nn.AdaptiveAvgPool2d(

                (1, 1)

            )

        )

        # =====================
        # SHARED EMBEDDING
        # =====================

        self.embedding = nn.Linear(

            128,

            embedding_dim,

        )

        # =====================
        # CLASSIFIER HEAD
        # =====================

        self.classifier = nn.Linear(

            embedding_dim,

            num_classes,

        )

    def forward(

        self,

        x,

        return_embedding=False,

    ):

        # =====================
        # BACKBONE
        # =====================

        x = self.backbone(

            x

        )

        x = self.global_pool(

            x

        )

        x = x.flatten(1)

        # =====================
        # EMBEDDING
        # =====================

        embedding = self.embedding(

            x

        )

        embedding = F.normalize(

            embedding,

            p=2,

            dim=1,

        )

        # =====================
        # CLASSIFICATION
        # =====================

        logits = self.classifier(

            embedding

        )

        if return_embedding:

            return logits, embedding

        return logits