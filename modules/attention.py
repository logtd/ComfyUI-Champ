import torch
from torch import nn
from einops import rearrange, repeat
from typing import Optional

from comfy.ldm.modules.diffusionmodules.util import AlphaBlender, timestep_embedding
from comfy.ldm.modules.attention import BasicTransformerBlock, exists

import comfy.ops
ops = comfy.ops.disable_weight_init



class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True, attn_precision=None, dtype=None, device=None, 
                 disable_temporal_crossattention=True,
                 num_norm_groups=1,
                 operations=ops):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = operations.GroupNorm(num_groups=num_norm_groups, num_channels=in_channels, eps=1e-6, affine=True, dtype=dtype, device=device)
        if not use_linear:
            self.proj_in = operations.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0, dtype=dtype, device=device)
        else:
            self.proj_in = operations.Linear(in_channels, inner_dim, dtype=dtype, device=device)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=0, disable_temporal_crossattention=disable_temporal_crossattention,
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint, attn_precision=attn_precision, dtype=dtype, device=device, operations=operations)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = operations.Conv2d(inner_dim,in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0, dtype=dtype, device=device)
        else:
            self.proj_out = operations.Linear(in_channels, inner_dim, dtype=dtype, device=device)
        self.use_linear = use_linear

    def forward(self, x, context=None, transformer_options={}):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context] * len(self.transformer_blocks)
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = x.movedim(1, 3).flatten(1, 2).contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            transformer_options["block_index"] = i
            x = block(x, context=context[i], transformer_options=transformer_options)
        if self.use_linear:
            x = self.proj_out(x)
        x = x.reshape(x.shape[0], h, w, x.shape[-1]).movedim(3, 1).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class SpatialVideoTransformer(SpatialTransformer):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        use_linear=False,
        context_dim=None,
        use_spatial_context=False,
        timesteps=None,
        merge_strategy: str = "fixed",
        merge_factor: float = 0.5,
        time_context_dim=None,
        ff_in=False,
        checkpoint=False,
        time_depth=1,
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        max_time_embed_period: int = 10000,
        attn_precision=None,
        dtype=None, device=None, operations=ops
    ):
        super().__init__(
            in_channels,
            n_heads,
            d_head,
            depth=depth,
            dropout=dropout,
            use_checkpoint=checkpoint,
            context_dim=context_dim,
            use_linear=use_linear,
            disable_self_attn=disable_self_attn,
            attn_precision=attn_precision,
            dtype=dtype, device=device, operations=operations
        )
        self.time_depth = time_depth
        self.depth = depth
        self.max_time_embed_period = max_time_embed_period

        time_mix_d_head = d_head
        n_time_mix_heads = n_heads

        time_mix_inner_dim = int(time_mix_d_head * n_time_mix_heads)

        inner_dim = n_heads * d_head
        if use_spatial_context:
            time_context_dim = context_dim

        self.time_stack = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_time_mix_heads,
                    time_mix_d_head,
                    dropout=dropout,
                    context_dim=time_context_dim,
                    # timesteps=timesteps,
                    checkpoint=checkpoint,
                    ff_in=ff_in,
                    inner_dim=time_mix_inner_dim,
                    disable_self_attn=disable_self_attn,
                    disable_temporal_crossattention=disable_temporal_crossattention,
                    attn_precision=attn_precision,
                    dtype=dtype, device=device, operations=operations
                )
                for _ in range(self.depth)
            ]
        )

        assert len(self.time_stack) == len(self.transformer_blocks)

        self.use_spatial_context = use_spatial_context
        self.in_channels = in_channels

        time_embed_dim = self.in_channels * 4
        self.time_pos_embed = nn.Sequential(
            operations.Linear(self.in_channels, time_embed_dim, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Linear(time_embed_dim, self.in_channels, dtype=dtype, device=device),
        )

        self.time_mixer = AlphaBlender(
            alpha=merge_factor, merge_strategy=merge_strategy
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        time_context: Optional[torch.Tensor] = None,
        timesteps: Optional[int] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        transformer_options={}
    ) -> torch.Tensor:
        _, _, h, w = x.shape
        x_in = x
        spatial_context = None
        if exists(context):
            spatial_context = context

        if self.use_spatial_context:
            assert (
                context.ndim == 3
            ), f"n dims of spatial context should be 3 but are {context.ndim}"

            if time_context is None:
                time_context = context
            time_context_first_timestep = time_context[::timesteps]
            time_context = repeat(
                time_context_first_timestep, "b ... -> (b n) ...", n=h * w
            )
        elif time_context is not None and not self.use_spatial_context:
            time_context = repeat(time_context, "b ... -> (b n) ...", n=h * w)
            if time_context.ndim == 2:
                time_context = rearrange(time_context, "b c -> b 1 c")

        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.use_linear:
            x = self.proj_in(x)

        num_frames = torch.arange(timesteps, device=x.device)
        num_frames = repeat(num_frames, "t -> b t", b=x.shape[0] // timesteps)
        num_frames = rearrange(num_frames, "b t -> (b t)")
        t_emb = timestep_embedding(num_frames, self.in_channels, repeat_only=False, max_period=self.max_time_embed_period).to(x.dtype)
        emb = self.time_pos_embed(t_emb)
        emb = emb[:, None, :]

        for it_, (block, mix_block) in enumerate(
            zip(self.transformer_blocks, self.time_stack)
        ):
            transformer_options["block_index"] = it_
            x = block(
                x,
                context=spatial_context,
                transformer_options=transformer_options,
            )

            x_mix = x
            x_mix = x_mix + emb

            B, S, C = x_mix.shape
            x_mix = rearrange(x_mix, "(b t) s c -> (b s) t c", t=timesteps)
            x_mix = mix_block(x_mix, context=time_context) #TODO: transformer_options
            x_mix = rearrange(
                x_mix, "(b s) t c -> (b t) s c", s=S, b=B // timesteps, c=C, t=timesteps
            )

            x = self.time_mixer(x_spatial=x, x_temporal=x_mix, image_only_indicator=image_only_indicator)

        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        if not self.use_linear:
            x = self.proj_out(x)
        out = x + x_in
        return out


