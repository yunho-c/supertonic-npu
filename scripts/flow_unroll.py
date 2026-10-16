# Attention computations adapted from ExecuTorch's Supertonic example.
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# See ../licenses/executorch.txt for the retained BSD license.
"""Explicit flow-step grouping and reuse of invariant Supertonic conditioning."""

import torch


class UnrolledFlow(torch.nn.Module):
    """One graph advances a shared estimator by ``unroll`` steps of a fixed schedule.

    Public arguments match the original estimator. The native caller must enforce
    total_step == schedule_steps and aligned, in-range current_step values.
    Intermediate latents are internal tensors, not graph outputs.
    """

    def __init__(self, estimator, unroll=2, schedule_steps=8,
                 reuse_conditioning=False, specialize_time=False):
        super().__init__()
        if unroll not in (1, 2, 4, 8) or schedule_steps < unroll or schedule_steps % unroll:
            raise ValueError("Unroll must be 1, 2, 4 or 8 and divide the fixed schedule")
        if specialize_time and not reuse_conditioning:
            raise ValueError("Time specialization requires conditioning reuse")
        self.estimator = estimator
        self.unroll = unroll
        self.schedule_steps = schedule_steps
        self.reuse_conditioning = reuse_conditioning
        self.specialize_time = specialize_time
        if specialize_time:
            field = estimator.model.vector_field
            with torch.no_grad():
                # Evaluate each timestep separately, with the same two CFG rows
                # as ordinary inference, to preserve the FP32 calculation order.
                times = [field.time_encoder(torch.full((2,), i / schedule_steps))
                         for i in range(schedule_steps)]
                for block in range(field.num_main_blocks):
                    linear = field.main_blocks[block * 6 + 1].linear
                    projections = [linear(time.transpose(1, 2)).transpose(1, 2)[:1]
                                   for time in times]
                    self.register_buffer(f"time_condition_{block}", torch.cat(projections).detach())

    @staticmethod
    def prepare_text(attn, text, latent_mask, text_mask):
        key = attn._split_heads(attn.W_key(text))
        value = attn._split_heads(attn.W_value(text))
        query_sine, query_cosine = attn._angles(latent_mask)
        key_sine, key_cosine = attn._angles(text_mask)
        key = attn._apply_rotary(key, key_sine, key_cosine)
        return (key, value, query_sine, query_cosine,
                text_mask[:, 0, :] != 0, latent_mask[:, 0, :] != 0)

    @staticmethod
    def text_condition(block, hidden, cache, mask):
        residual = hidden * mask
        inputs = residual.transpose(1, 2)
        attn = block.attn
        key, value, sine, cosine, valid_keys, valid_queries = cache
        query = attn._apply_rotary(attn._split_heads(attn.W_query(inputs)), sine, cosine)
        scores = attn._scaled_scores(query, key)
        scores = scores.masked_fill(~valid_keys.unsqueeze(0).unsqueeze(2), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(valid_queries.unsqueeze(0).unsqueeze(-1), weights, torch.zeros_like(weights))
        attended = torch.matmul(weights, value).permute(1, 2, 0, 3).reshape(inputs.shape[0], inputs.shape[1], -1)
        attended = attn.out_fc(attended) * mask.transpose(1, 2)
        return block.norm(residual + attended.transpose(1, 2)) * mask

    @staticmethod
    def prepare_style(attn, key, value):
        return (torch.tanh(attn._split_heads(attn.W_key(key)).transpose(-2, -1)),
                attn._split_heads(attn.W_value(value)))

    @staticmethod
    def style_condition(block, hidden, cache, mask):
        residual = hidden * mask
        attn = block.attention
        key, value = cache
        query = attn._split_heads(attn.W_query(residual.transpose(1, 2)))
        weights = torch.softmax(torch.matmul(query, key) / attn.score_scale, dim=-1)
        weights = torch.where(mask.transpose(1, 2).unsqueeze(0) != 0, weights, torch.zeros_like(weights))
        attended = torch.matmul(weights, value)
        attended = torch.cat(torch.unbind(attended, dim=0), dim=-1)
        attended = attn.out_fc(attended) * mask.transpose(1, 2)
        return block.norm(residual + attended.transpose(1, 2)) * mask

    def forward(self, noisy_latent, text_emb, style_ttl, latent_mask, text_mask, current_step, total_step):
        if not self.reuse_conditioning:
            for offset in range(self.unroll):
                noisy_latent = self.estimator(noisy_latent, text_emb, style_ttl, latent_mask,
                                              text_mask, current_step + offset, total_step)
            return noisy_latent

        model = self.estimator.model
        field = model.vector_field
        batch = noisy_latent.shape[0]
        text = torch.cat((text_emb, model.uncond_masker.text_special_token.expand(batch, -1, text_emb.shape[2])), dim=0)
        style_key = torch.cat((model.style_key.expand(batch, -1, -1),
                               model.uncond_masker.style_key_special_token.expand(batch, -1, -1)), dim=0)
        style_value = torch.cat((style_ttl, model.uncond_masker.style_value_special_token.expand(batch, -1, -1)), dim=0)
        lm, tm = latent_mask.repeat(2, 1, 1), text_mask.repeat(2, 1, 1)
        text_cache = [self.prepare_text(field.main_blocks[i * 6 + 3].attn, text.transpose(1, 2), lm, tm)
                      for i in range(field.num_main_blocks)]
        style_cache = [self.prepare_style(field.main_blocks[i * 6 + 5].attention, style_key, style_value)
                       for i in range(field.num_main_blocks)]
        step_size = 1.0 / self.schedule_steps if self.specialize_time else torch.reciprocal(total_step).reshape(-1, 1, 1)
        for offset in range(self.unroll):
            hidden = field.proj_in(noisy_latent.repeat(2, 1, 1)) * lm
            if not self.specialize_time:
                time = field.time_encoder(((current_step + offset) / total_step).repeat(2))
            for block in range(field.num_main_blocks):
                index = block * 6
                hidden = field.main_blocks[index](hidden, lm)
                if self.specialize_time:
                    table = getattr(self, f"time_condition_{block}")
                    # Full unrolling needs no runtime indexing. Partial groups
                    # select their starting point from the fixed schedule table.
                    condition = (table[offset:offset + 1] if self.unroll == self.schedule_steps else
                                 torch.index_select(table, 0, current_step.to(torch.int64) + offset))
                    hidden = (hidden + condition) * lm
                else:
                    hidden = field.main_blocks[index + 1](hidden, time, lm)
                hidden = field.main_blocks[index + 2](hidden, lm)
                hidden = self.text_condition(field.main_blocks[index + 3], hidden, text_cache[block], lm)
                hidden = field.main_blocks[index + 4](hidden, lm)
                hidden = self.style_condition(field.main_blocks[index + 5], hidden, style_cache[block], lm)
            vector = field.proj_out(field.last_convnext(hidden, lm)) * lm
            conditional, unconditional = vector.chunk(2, dim=0)
            noisy_latent = (noisy_latent + step_size * (4.0 * conditional - 3.0 * unconditional)) * latent_mask
        return noisy_latent
