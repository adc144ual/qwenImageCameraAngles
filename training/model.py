import logging

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)


class QwenSingleGPUWrapper(nn.Module):
    """Wrapper de QwenImageTransformer2DModel para entrenamiento en 1 GPU."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            self.inner_model = model.base_model.model
        else:
            self.inner_model = model

        total_layers = len(self.inner_model.transformer_blocks)
        logger.info(f"QwenSingleGPUWrapper: {total_layers} bloques transformer (sin split).")

    def _block_checkpoint(self, block, h, e, mask, temb, r0, r1):
        def _fwd(h, e, mask, temb, r0, r1):
            if r0.numel() > r1.numel():
                img_rot, txt_rot = r0, r1
            else:
                img_rot, txt_rot = r1, r0
            return block(
                hidden_states=h,
                encoder_hidden_states=e,
                encoder_hidden_states_mask=mask,
                temb=temb,
                image_rotary_emb=(img_rot, txt_rot)
            )
        return checkpoint(_fwd, h, e, mask, temb, r0, r1, use_reentrant=False)

    def forward(self, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, timestep):
        hidden_states = self.inner_model.img_in(hidden_states)
        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.inner_model.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.inner_model.txt_in(encoder_hidden_states)

        temb = self.inner_model.time_text_embed(timestep, hidden_states)
        temb = temb.to(dtype=hidden_states.dtype)

        B = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        grid_sq = int(seq_len ** 0.5)
        if grid_sq * grid_sq == seq_len:
            img_shapes = [[(1, grid_sq, grid_sq)]] * B
        else:
            half = seq_len // 2
            grid_sq = int(half ** 0.5)
            img_shapes = [[(2, grid_sq, grid_sq)]] * B

        full_len = encoder_hidden_states.shape[1]

        image_rotary_emb = self.inner_model.pos_embed(
            img_shapes,
            max_txt_seq_len=full_len,
            device=hidden_states.device
        )
        r0, r1 = image_rotary_emb

        for block in self.inner_model.transformer_blocks:
            if self.training:
                if not hidden_states.requires_grad:
                    hidden_states.requires_grad_(True)
                if not encoder_hidden_states.requires_grad:
                    encoder_hidden_states.requires_grad_(True)
                encoder_hidden_states, hidden_states = self._block_checkpoint(
                    block, hidden_states, encoder_hidden_states,
                    encoder_hidden_states_mask, temb, r0, r1,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=(r0, r1),
                )

        hidden_states = self.inner_model.norm_out(hidden_states, temb)
        hidden_states = self.inner_model.proj_out(hidden_states)

        return hidden_states