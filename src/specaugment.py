import random

import torch
import torch.nn as nn


class SpecAugment(nn.Module):

    def __init__(

        self,

        freq_mask_param=12,

        time_mask_param=25,

        num_freq_masks=2,

        num_time_masks=2,

        p=0.8,

    ):

        super().__init__()

        self.freq_mask_param = freq_mask_param

        self.time_mask_param = time_mask_param

        self.num_freq_masks = num_freq_masks

        self.num_time_masks = num_time_masks

        self.p = p


    def forward(self, x):

        """
        Input
        -----

        x: [B,1,80,250]

        """

        if not self.training:

            return x

        if random.random() > self.p:

            return x

        x = x.clone()

        B, C, F, T = x.shape

        # ==========================
        # Frequency masking
        # ==========================

        for _ in range(self.num_freq_masks):

            mask_width = random.randint(

                0,

                self.freq_mask_param,

            )

            if mask_width == 0:

                continue

            start = random.randint(

                0,

                max(0, F - mask_width),

            )

            x[:, :, start:start + mask_width, :] = 0

        # ==========================
        # Time masking
        # ==========================

        for _ in range(self.num_time_masks):

            mask_width = random.randint(

                0,

                self.time_mask_param,

            )

            if mask_width == 0:

                continue

            start = random.randint(

                0,

                max(0, T - mask_width),

            )

            x[:, :, :, start:start + mask_width] = 0

        return x