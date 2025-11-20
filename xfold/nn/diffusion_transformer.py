# Copyright 2024 xfold authors
# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md


from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from xfold.nn import atom_layout
from xfold import fastnn


class AdaptiveLayerNorm(nn.Module):
    def __init__(self,
                 c_x: int,
                 c_single_cond: int,
                 use_single_cond: bool = False) -> None:

        super(AdaptiveLayerNorm, self).__init__()

        self.c_x = c_x
        self.c_single_cond = c_single_cond
        self.use_single_cond = use_single_cond

        if self.use_single_cond is True:
            self.layer_norm = fastnn.LayerNorm(
                self.c_x, elementwise_affine=False, bias=False)
            self.single_cond_layer_norm = fastnn.LayerNorm(
                self.c_single_cond, bias=False)
            self.single_cond_scale = nn.Linear(
                self.c_single_cond, self.c_x, bias=True)
            self.single_cond_bias = nn.Linear(
                self.c_single_cond, self.c_x, bias=False)
        else:
            self.layer_norm = fastnn.LayerNorm(self.c_x)

    def forward(self,
                x: torch.Tensor,
                single_cond: Optional[torch.Tensor] = None,
                precomputed_cond: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:

        assert (single_cond is None and precomputed_cond is None) == (self.use_single_cond is False)

        if self.use_single_cond is True:
            x = self.layer_norm(x)
            if precomputed_cond is not None:
                single_scale, single_bias = precomputed_cond
            else:
                single_cond = self.single_cond_layer_norm(single_cond)
                single_scale = self.single_cond_scale(single_cond)
                single_bias = self.single_cond_bias(single_cond)
            return torch.sigmoid(single_scale) * x + single_bias
        else:
            return self.layer_norm(x)


class AdaLNZero(nn.Module):
    def __init__(self,
                 c_in: int,
                 c_out: int,
                 c_single_cond: int,
                 use_single_cond: bool = False) -> None:
        super(AdaLNZero, self).__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.c_single_cond = c_single_cond
        self.use_single_cond = use_single_cond

        self.transition2 = nn.Linear(self.c_in, self.c_out, bias=False)
        if self.use_single_cond is True:
            self.adaptive_zero_cond = nn.Linear(
                self.c_single_cond, self.c_out, bias=True)

    def forward(self,
                x: torch.Tensor,
                single_cond: Optional[torch.Tensor] = None,
                precomputed_cond: Optional[torch.Tensor] = None) -> torch.Tensor:

        assert (single_cond is None and precomputed_cond is None) == (self.use_single_cond is False)

        output = self.transition2(x)
        if self.use_single_cond is True:
            if precomputed_cond is not None:
                cond = precomputed_cond
            else:
                cond = self.adaptive_zero_cond(single_cond)
            output = torch.sigmoid(cond) * output
        return output


class DiffusionTransition(nn.Module):
    def __init__(self,
                 c_x: int,
                 c_single_cond: int,
                 num_intermediate_factor: int = 2,
                 use_single_cond: bool = False) -> None:
        super(DiffusionTransition, self).__init__()

        self.c_x = c_x
        self.c_single_cond = c_single_cond
        self.num_intermediate_factor = num_intermediate_factor
        self.use_single_cond = use_single_cond

        self.adaptive_layernorm = AdaptiveLayerNorm(
            self.c_x, self.c_single_cond, self.use_single_cond)
        self.transition1 = nn.Linear(
            self.c_x, 2 * self.c_x * self.num_intermediate_factor, bias=False)

        self.adaptive_zero_init = AdaLNZero(
            self.num_intermediate_factor * self.c_x,
            self.c_x,
            self.c_single_cond,
            self.use_single_cond
        )

    def forward(self, x: torch.Tensor, single_cond: Optional[torch.Tensor] = None,
                precomputed_cond: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]] = None) -> torch.Tensor:
        
        norm_cond = None
        zero_cond = None
        if precomputed_cond is not None:
            norm_cond, zero_cond = precomputed_cond

        x = self.adaptive_layernorm(x, single_cond, precomputed_cond=norm_cond)
        c = fastnn.gated_linear_unit(x, self.transition1.weight.T)
        return self.adaptive_zero_init(c, single_cond, precomputed_cond=zero_cond)


class SelfAttention(nn.Module):
    def __init__(self,
                 c_x: int = 768,
                 c_single_cond: int = 384,
                 num_head: int = 16,
                 use_single_cond: bool = False) -> None:

        super(SelfAttention, self).__init__()

        self.c_x = c_x
        self.c_single_cond = c_single_cond
        self.num_head = num_head

        self.qkv_dim = self.c_x // self.num_head
        self.use_single_cond = use_single_cond

        self.adaptive_layernorm = AdaptiveLayerNorm(
            self.c_x, self.c_single_cond, self.use_single_cond)

        self.q_projection = nn.Linear(self.c_x, self.c_x, bias=True)
        self.k_projection = nn.Linear(self.c_x, self.c_x, bias=False)
        self.v_projection = nn.Linear(self.c_x, self.c_x, bias=False)

        self.gating_query = nn.Linear(self.c_x, self.c_x, bias=False)

        self.adaptive_zero_init = AdaLNZero(
            self.c_x, self.c_x, self.c_single_cond, self.use_single_cond)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                pair_logits: Optional[torch.Tensor] = None,
                single_cond: Optional[torch.Tensor] = None,
                precomputed_cond: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (num_tokens, ch)
            mask (torch.Tensor): (num_tokens,)
            pair_logits (torch.Tensor, optional): (num_heads, num_tokens, num_tokens)
        """

        assert (single_cond is None and precomputed_cond is None) == (self.use_single_cond is False)

        norm_cond = None
        zero_cond = None
        if precomputed_cond is not None:
            norm_cond, zero_cond = precomputed_cond

        x = self.adaptive_layernorm(x, single_cond, precomputed_cond=norm_cond)

        # Fuse q/k/v projections into a single GEMM on x to reduce
        # the number of large matrix multiplications.
        n, c = x.shape
        w_q = self.q_projection.weight  # [c, c]
        w_k = self.k_projection.weight  # [c, c]
        w_v = self.v_projection.weight  # [c, c]
        w_qkv = torch.cat([w_q, w_k, w_v], dim=0)  # [3c, c]

        qkv = torch.matmul(x, w_qkv.transpose(0, 1))  # [n, 3c]
        q, k, v = qkv.split(c, dim=-1)

        if self.q_projection.bias is not None:
            q = q + self.q_projection.bias

        # [n, c] -> [1, h, n, c//h]
        q = q.view(n, self.num_head, self.qkv_dim).permute(1, 0, 2).unsqueeze(0).contiguous()
        k = k.view(n, self.num_head, self.qkv_dim).permute(1, 0, 2).unsqueeze(0).contiguous()
        v = v.view(n, self.num_head, self.qkv_dim).permute(1, 0, 2).unsqueeze(0).contiguous()

        weighted_avg = fastnn.dot_product_attention(
            q, k, v, mask=mask, bias=pair_logits
        )

        weighted_avg = weighted_avg.squeeze(0)
        weighted_avg = einops.rearrange(weighted_avg, 'h q c -> q (h c)')

        gate_logits = self.gating_query(x)
        weighted_avg *= torch.sigmoid(gate_logits)

        return self.adaptive_zero_init(weighted_avg, single_cond, precomputed_cond=zero_cond)


class DiffusionTransformer(nn.Module):
    def __init__(self,
                 c_act: int = 768,
                 c_single_cond: int = 384,
                 c_pair_cond: int = 128,
                 num_head: int = 16,
                 num_blocks: int = 24,
                 super_block_size: int = 4) -> None:

        super(DiffusionTransformer, self).__init__()

        self.c_act = c_act
        self.c_single_cond = c_single_cond
        self.c_pair_cond = c_pair_cond
        self.num_head = num_head
        self.num_blocks = num_blocks
        self.super_block_size = super_block_size

        self.num_super_blocks = self.num_blocks // self.super_block_size

        self.pair_input_layer_norm = fastnn.LayerNorm(self.c_pair_cond)
        self.pair_logits_projection = nn.ModuleList(
            [nn.Linear(self.c_pair_cond, self.super_block_size * self.num_head, bias=False) for _ in range(self.num_super_blocks)])

        self.self_attention = nn.ModuleList(
            [SelfAttention(self.c_act, self.c_single_cond, use_single_cond=True) for _ in range(self.num_blocks)])
        self.transition_block = nn.ModuleList(
            [DiffusionTransition(self.c_act, self.c_single_cond, use_single_cond=True) for _ in range(self.num_blocks)])

        # Cache for pair_logits to avoid recomputation across diffusion steps
        self._pair_logits_cache_key = None
        self._pair_logits_cache = None

        # Cache for single_cond to avoid recomputation
        self._single_cond_cache_key = None
        self._single_cond_cache = None

    def _compute_pair_logits(self, pair_cond: torch.Tensor) -> list[torch.Tensor]:
        """Compute (or reuse cached) pair_logits for all super blocks.

        pair_cond does not depend on the diffusion noise level, so across
        diffusion steps we can reuse the same pair_logits as long as the
        pair_cond tensor object is the same.
        """
        cache_key = id(pair_cond)
        if cache_key == self._pair_logits_cache_key and self._pair_logits_cache is not None:
            return self._pair_logits_cache

        pair_act = self.pair_input_layer_norm(pair_cond)

        pair_logits_all: list[torch.Tensor] = []
        for super_block_i in range(self.num_super_blocks):
            pair_logits = self.pair_logits_projection[super_block_i](pair_act)
            pair_logits = einops.rearrange(
                pair_logits, 'n s (b h) -> b h n s', h=self.num_head
            )
            pair_logits_all.append(pair_logits)

        self._pair_logits_cache_key = cache_key
        self._pair_logits_cache = pair_logits_all
        return pair_logits_all

    def _compute_single_cond_cache(self, single_cond: torch.Tensor) -> list[Tuple]:
        """Compute (or reuse cached) single_cond projections for all blocks."""
        cache_key = id(single_cond)
        if cache_key == self._single_cond_cache_key and self._single_cond_cache is not None:
            return self._single_cond_cache

        cache = []
        for i in range(self.num_blocks):
            # Self Attention
            sa = self.self_attention[i]
            # Adaptive LayerNorm
            sc = sa.adaptive_layernorm.single_cond_layer_norm(single_cond)
            sa_norm = (sa.adaptive_layernorm.single_cond_scale(sc),
                       sa.adaptive_layernorm.single_cond_bias(sc))
            # AdaLNZero
            sa_zero = sa.adaptive_zero_init.adaptive_zero_cond(single_cond)
            
            # Transition
            tr = self.transition_block[i]
            # Adaptive LayerNorm
            sc = tr.adaptive_layernorm.single_cond_layer_norm(single_cond)
            tr_norm = (tr.adaptive_layernorm.single_cond_scale(sc),
                       tr.adaptive_layernorm.single_cond_bias(sc))
            # AdaLNZero
            tr_zero = tr.adaptive_zero_init.adaptive_zero_cond(single_cond)
            
            cache.append(((sa_norm, sa_zero), (tr_norm, tr_zero)))

        self._single_cond_cache_key = cache_key
        self._single_cond_cache = cache
        return cache

    def forward(self,
                act: torch.Tensor,
                mask: torch.Tensor,
                single_cond: torch.Tensor,
                pair_cond:  torch.Tensor):

        single_cond_cache = self._compute_single_cond_cache(single_cond)

        for super_block_i in range(self.num_super_blocks):
            pair_logits = pair_logits_all[super_block_i]
            for j in range(self.super_block_size):
                block_idx = super_block_i * self.super_block_size + j
                sa_cond, tr_cond = single_cond_cache[block_idx]

                act += self.self_attention[block_idx](
                    act, mask, pair_logits[j, ...], single_cond, precomputed_cond=sa_cond)
                act += self.transition_block[block_idx](act, single_cond, precomputed_cond=tr_cond)

        return act


class CrossAttention(nn.Module):
    def __init__(self, key_dim: int = 128, value_dim: int = 128, c_single_cond: int = 128, num_head: int = 4) -> None:
        super(CrossAttention, self).__init__()

        self.key_dim = key_dim
        self.value_dim = value_dim
        self.c_single_cond = c_single_cond
        self.num_head = num_head

        self.key_dim_per_head = self.key_dim // self.num_head
        self.value_dim_per_head = self.value_dim // self.num_head

        self.q_scale = self.key_dim_per_head ** (-0.5)

        self.q_adaptive_layernorm = AdaptiveLayerNorm(
            c_x=self.key_dim, c_single_cond=self.c_single_cond, use_single_cond=True)
        self.k_adaptive_layernorm = AdaptiveLayerNorm(
            c_x=self.key_dim, c_single_cond=self.c_single_cond, use_single_cond=True)

        self.q_projection = nn.Linear(self.key_dim, self.key_dim, bias=True)
        self.k_projection = nn.Linear(self.key_dim, self.key_dim, bias=False)
        self.v_projection = nn.Linear(
            self.value_dim, self.value_dim, bias=False)

        self.gating_query = nn.Linear(self.key_dim, self.value_dim, bias=False)
        self.adaptive_zero_init = AdaLNZero(
            self.value_dim, self.value_dim, self.key_dim, use_single_cond=True)

    def forward(
        self,
        x_q: torch.Tensor,
        x_k: torch.Tensor,
        mask_q: torch.Tensor,
        mask_k: torch.Tensor,
        pair_logits: Optional[torch.Tensor] = None,
        single_cond_q: Optional[torch.Tensor] = None,
        single_cond_k: Optional[torch.Tensor] = None,
        precomputed_cond: Optional[Tuple[
            Tuple[torch.Tensor, torch.Tensor], # q_norm
            Tuple[torch.Tensor, torch.Tensor], # k_norm
            torch.Tensor # zero_init
        ]] = None
    ) -> torch.Tensor:
        assert len(mask_q.shape) == len(x_q.shape) - \
            1, f'{mask_q.shape}, {x_q.shape}'
        assert len(mask_k.shape) == len(x_k.shape) - \
            1, f'{mask_k.shape}, {x_k.shape}'

        bias = (
            1e9
            * mask_q.logical_not()[..., None, :, None]
            * mask_k.logical_not()[..., None, None, :]
        )

        q_norm_cond = None
        k_norm_cond = None
        zero_cond = None
        if precomputed_cond is not None:
            q_norm_cond, k_norm_cond, zero_cond = precomputed_cond

        x_q = self.q_adaptive_layernorm(x_q, single_cond_q, precomputed_cond=q_norm_cond)
        x_k = self.k_adaptive_layernorm(x_k, single_cond_k, precomputed_cond=k_norm_cond)

        # q from x_q, k/v from x_k. Fuse k/v projections into a single GEMM.
        q = self.q_projection(x_q)

        # x_k last dim should match both key_dim and value_dim in practice.
        w_k = self.k_projection.weight  # [key_dim, key_dim]
        w_v = self.v_projection.weight  # [value_dim, value_dim]
        if self.key_dim == self.value_dim:
            w_kv = torch.cat([w_k, w_v], dim=0)  # [2C, C]
            kv = torch.matmul(x_k, w_kv.transpose(0, 1))  # [..., 2C]
            k, v = kv.split(self.key_dim, dim=-1)
        else:
            k = self.k_projection(x_k)
            v = self.v_projection(x_k)

        q = torch.reshape(q, q.shape[:-1] +
                          (self.num_head, self.key_dim_per_head))
        k = torch.reshape(k, k.shape[:-1] +
                          (self.num_head, self.key_dim_per_head))
        v = torch.reshape(v, v.shape[:-1] +
                          (self.num_head, self.value_dim_per_head))

        # Use SDPA
        # q, k, v shape: (..., num_head, dim)
        # SDPA expects: (batch, num_head, seq_len, dim)
        # Here we have (batch, seq_len, num_head, dim)
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        # bias shape: (..., 1, Nq, Nk)
        # pair_logits shape: (..., H, Nq, Nk)
        attn_bias = bias
        if pair_logits is not None:
            attn_bias = attn_bias + pair_logits

        weighted_avg = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias
        )
        
        # Output: (batch, num_head, seq_len, dim) -> (batch, seq_len, num_head, dim)
        weighted_avg = weighted_avg.transpose(-2, -3)
        
        weighted_avg = torch.reshape(
            weighted_avg, weighted_avg.shape[:-2] + (-1,))

        gate_logits = self.gating_query(x_q)
        weighted_avg *= torch.sigmoid(gate_logits)

        return self.adaptive_zero_init(weighted_avg, single_cond_q, precomputed_cond=zero_cond)


class DiffusionCrossAttTransformer(nn.Module):
    def __init__(self, c_query: int = 128, c_single_cond: int = 128, c_pair_cond: int = 16, num_blocks: int = 3, num_head: int = 4) -> None:
        super(DiffusionCrossAttTransformer, self).__init__()

        self.c_query = c_query
        self.c_single_cond = c_single_cond
        self.c_pair_cond = c_pair_cond

        self.num_blocks = num_blocks
        self.num_head = num_head

        self.pair_input_layer_norm = fastnn.LayerNorm(self.c_pair_cond, bias=False)
        self.pair_logits_projection = nn.Linear(
            self.c_pair_cond, self.num_blocks * self.num_head, bias=False)

        self.cross_attention = nn.ModuleList(
            [CrossAttention(num_head=self.num_head) for _ in range(self.num_blocks)])

            [DiffusionTransition(c_x=self.c_query, c_single_cond=self.c_single_cond, use_single_cond=True) for _ in range(self.num_blocks)])

        self._single_cond_cache_key = None
        self._single_cond_cache = None

    def _compute_single_cond_cache(self, single_cond_q: torch.Tensor, single_cond_k: torch.Tensor) -> list[Tuple]:
        cache_key = (id(single_cond_q), id(single_cond_k))
        if cache_key == self._single_cond_cache_key and self._single_cond_cache is not None:
            return self._single_cond_cache
        
        cache = []
        for i in range(self.num_blocks):
            # Cross Attention
            ca = self.cross_attention[i]
            # Q Norm
            sc_q = ca.q_adaptive_layernorm.single_cond_layer_norm(single_cond_q)
            q_norm = (ca.q_adaptive_layernorm.single_cond_scale(sc_q),
                      ca.q_adaptive_layernorm.single_cond_bias(sc_q))
            # K Norm
            sc_k = ca.k_adaptive_layernorm.single_cond_layer_norm(single_cond_k)
            k_norm = (ca.k_adaptive_layernorm.single_cond_scale(sc_k),
                      ca.k_adaptive_layernorm.single_cond_bias(sc_k))
            # Zero Init
            zero_cond = ca.adaptive_zero_init.adaptive_zero_cond(single_cond_q)
            
            # Transition
            tr = self.transition_block[i]
            # Norm
            sc_tr = tr.adaptive_layernorm.single_cond_layer_norm(single_cond_q)
            tr_norm = (tr.adaptive_layernorm.single_cond_scale(sc_tr),
                       tr.adaptive_layernorm.single_cond_bias(sc_tr))
            # Zero
            tr_zero = tr.adaptive_zero_init.adaptive_zero_cond(single_cond_q)
            
            cache.append(((q_norm, k_norm, zero_cond), (tr_norm, tr_zero)))
            
        self._single_cond_cache_key = cache_key
        self._single_cond_cache = cache
        return cache

    def forward(
        self,
        queries_act: torch.Tensor,  # (num_subsets, num_queries, ch)
        queries_mask: torch.Tensor,  # (num_subsets, num_queries)
        queries_to_keys: atom_layout.GatherInfo,  # (num_subsets, num_keys)
        keys_mask: torch.Tensor,  # (num_subsets, num_keys)
        queries_single_cond: torch.Tensor,  # (num_subsets, num_queries, ch)
        keys_single_cond: torch.Tensor,  # (num_subsets, num_keys, ch)
        pair_cond: torch.Tensor,  # (num_subsets, num_queries, num_keys, ch)
    ) -> torch.Tensor:

        pair_act = self.pair_input_layer_norm(pair_cond)
        pair_logits = self.pair_logits_projection(pair_act)

        pair_logits = einops.rearrange(
            pair_logits, 'n q k (b h) -> b n h q k', h=self.num_head)

        single_cond_cache = self._compute_single_cond_cache(queries_single_cond, keys_single_cond)

        for block_idx in range(self.num_blocks):
            keys_act = atom_layout.convert(
                queries_to_keys, queries_act, layout_axes=(-3, -2)
            )

            ca_cond, tr_cond = single_cond_cache[block_idx]

            queries_act += self.cross_attention[block_idx](
                x_q=queries_act,
                x_k=keys_act,
                mask_q=queries_mask,
                mask_k=keys_mask,
                pair_logits=pair_logits[block_idx,...],
                single_cond_q=queries_single_cond,
                single_cond_k=keys_single_cond,
                precomputed_cond=ca_cond
            )
            queries_act += self.transition_block[block_idx](
                queries_act,
                queries_single_cond,
                precomputed_cond=tr_cond
            )

        return queries_act
