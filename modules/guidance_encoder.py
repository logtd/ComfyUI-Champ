from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops
ops = comfy.ops.disable_weight_init
from .attention import SpatialTransformer


class GuidanceEncoder(nn.Module):
    def __init__(
        self,
        guidance_embedding_channels: int,
        guidance_input_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 96, 256),
        attention_num_heads: int = 8,
    ):
        super().__init__()
        self.guidance_input_channels = guidance_input_channels
        self.conv_in = ops.Conv2d(
            guidance_input_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList([])
        self.attentions = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]

            self.blocks.append(
                ops.Conv2d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.attentions.append(
                SpatialTransformer(
                    channel_in,
                    attention_num_heads,
                    channel_in // attention_num_heads,
                    disable_temporal_crossattention=True,
                    num_norm_groups=1,
                )
            )

            self.blocks.append(
                ops.Conv2d(
                    channel_in, channel_out, kernel_size=3, padding=1, stride=2
                )
            )
            self.attentions.append(
                SpatialTransformer(
                    channel_out,
                    attention_num_heads,
                    channel_out // attention_num_heads,
                    disable_temporal_crossattention=True,
                    num_norm_groups=32,
                )
            )

        # This is not in the given Champ checkpoint
        # attention_channel_out = block_out_channels[-1]
        # self.guidance_attention = SpatialTransformer(
        #     attention_channel_out,
        #     attention_num_heads,
        #     attention_channel_out // attention_num_heads,
        #     disable_temporal_crossattention=True,
        #     num_norm_groups=32,
        # )

        self.conv_out = ops.Conv2d(
            block_out_channels[-1],
            guidance_embedding_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, condition):
        embedding = self.conv_in(condition)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        # FIXME: Temporarily only use the last attention. --> This was in Champ code
        embedding = self.attentions[-1](embedding)
        embedding = self.conv_out(embedding)

        return embedding