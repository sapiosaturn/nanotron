"""
DeepSeek V3 nanotron model for training and inference, in PyTorch.

TODO:
- fix tensor parallel for MLA
- add expert parallel based on moe example
- look into FlashMLA integration
- fix load balancing
"""
import math
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import CheckpointFunction

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import Config, DeepSeekV3Config, ParallelismArgs
from nanotron.config.models_config import RandomInit, SpectralMupInit
from nanotron.generation.generate_store import AttachableStore
from nanotron.logging import log_rank
from nanotron.models import NanotronModel
from nanotron.nn.activations import ACT2FN
from nanotron.nn.layer_norm import TritonRMSNorm
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.pipeline_parallel.block import PipelineBlock, TensorPointer
from nanotron.parallel.pipeline_parallel.p2p import P2P
from nanotron.parallel.tensor_parallel.functional import sharded_cross_entropy
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelLinearMode,
    TensorParallelRowLinear,
)
from nanotron.random import RandomStates
from nanotron.scaling.parametrization import SpectralMupParametrizator, StandardParametrizator
from nanotron.utils import checkpoint_method

logger = logging.get_logger(__name__)

def precompute_freqs_cis(config: DeepSeekV3Config) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = config.qk_rope_head_dim
    seqlen = config.seq_len
    beta_fast = config.beta_fast
    beta_slow = config.beta_slow
    base = config.rope_theta
    factor = config.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > config.seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, config.seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:x.size(1)].view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)

class Embedding(nn.Module, AttachableStore):
    def __init__(self, tp_pg: dist.ProcessGroup, config: DeepSeekV3Config, parallel_config: Optional[ParallelismArgs]):
        super().__init__()
        self.token_embedding = TensorParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
            pg=tp_pg,
            mode=parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE,
        )
        self.pg = tp_pg

    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor):  # [batch_size, seq_length]
        # Format input in `[seq_length, batch_size]` to support high TP with low batch_size
        # input_ids = input_ids.transpose(0, 1)
        # for now, we stay with [batch_size, seq_length]
        input_embeds = self.token_embedding(input_ids)
        return {"input_embeds": input_embeds} # we hold! [batch_size, seq_length, hidden_size]

# taken from llama model implementation
class CoreAttention(nn.Module):
    def __init__(self, config: DeepSeekV3Config, parallel_config: Optional[ParallelismArgs], layer_idx: int):
        super().__init__()
        assert (
            config.hidden_size % config.num_attention_heads == 0
        ), f"Hidden size {config.hidden_size} must be divisible by number of attention heads {config.num_attention_heads}."
        self.d_qk = config.hidden_size // config.num_attention_heads
        self.d_v = config.hidden_size // config.num_attention_heads
        self.is_using_mup = config.is_using_mup
        self.checkpoint_attention = False  # Because flash_attn already does checkpointing

    @checkpoint_method(attr_name="checkpoint_attention")
    def forward(
        self,
        query_states: torch.Tensor,  # [batch_size * q_length, n_local_q_heads, inner_dim]
        key_states: torch.Tensor,  # [batch_size * kv_length, n_local_kv_heads, inner_dim]
        value_states: torch.Tensor,  # [batch_size * kv_length, n_local_kv_heads, inner_dim]
        q_sequence_mask: torch.Tensor,  # torch.BoolTensor [batch_size, q_length] (can be broadcasted to that size)
        kv_sequence_mask: torch.Tensor,  # torch.BoolTensor [batch_size, kv_length] (can be broadcasted to that size)
    ):
        from flash_attn.flash_attn_interface import flash_attn_varlen_func

        # TODO @thomasw21: Compute once, instead of computing for each layers.
        cu_seqlens_q = torch.zeros((q_sequence_mask.shape[0] + 1), dtype=torch.int32, device=query_states.device)
        cu_seqlens_k = torch.zeros((kv_sequence_mask.shape[0] + 1), dtype=torch.int32, device=query_states.device)
        torch.cumsum(q_sequence_mask.sum(-1, dtype=torch.int32), dim=0, dtype=torch.int32, out=cu_seqlens_q[1:])
        torch.cumsum(kv_sequence_mask.sum(-1, dtype=torch.int32), dim=0, dtype=torch.int32, out=cu_seqlens_k[1:])

        # TODO(kunhao): flash attn's causal means that the query can only attend to the keys before it. This is not
        # what we want if we are using kv cache. This is a hack as we always have q_length == 1 when using kv cache.
        causal = False if q_sequence_mask.shape[1] == 1 else True

        # NOTE: this scale is for µTransfer,
        # in SP, we use sqrt(1/d_h)
        softmax_scale = 1 / query_states.shape[-1] if self.is_using_mup else None
        attn_output = flash_attn_varlen_func(
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=q_sequence_mask.shape[1],
            max_seqlen_k=kv_sequence_mask.shape[1],
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal,
            return_attn_probs=False,
        )

        return attn_output

class DeepSeekV3MLA(nn.Module, AttachableStore):
    def __init__(
        self,
        config: DeepSeekV3Config,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_idx: int,
    ):

        super().__init__()
        # Tensor parallel considerations: We split tensors along head dimension
        assert (
            config.num_attention_heads % tp_pg.size() == 0
        ), f"Number of attention heads ({config.num_attention_heads}) must be divisible by TP size ({tp_pg.size()})."
        try:
            assert (
                config.num_key_value_heads % tp_pg.size() == 0
            ), f"Number of key/value heads ({config.num_key_value_heads}) must be divisible by TP size ({tp_pg.size()})."
        except AttributeError:
            log_rank(
                "WARNING: num_key_value_heads not defined, assuming it is equal to num_attention_heads",
                logger=logger,
                level=logging.WARNING,
                rank=0,
            )
            # If num_key_value_heads is not defined, we assume that it is equal to num_attention_heads
            config.num_key_value_heads = config.num_attention_heads
        assert (
            config.num_attention_heads % config.num_key_value_heads == 0
        ), f"Number of attention heads ({config.num_attention_heads}) must be divisible by number of key/value heads ({config.num_key_value_heads})."
        # self.n_local_q_heads = config.num_attention_heads // tp_pg.size()
        # self.n_local_kv_heads = config.num_key_value_heads // tp_pg.size()
        self.n_local_q_heads = config.num_attention_heads
        self.n_local_kv_heads = config.num_key_value_heads
        self.n_repeats = config.num_attention_heads // config.num_key_value_heads
        self.is_gqa = config.num_attention_heads != config.num_key_value_heads  # Whether we are using GQA or not

        # MLA specific params
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.d_qk = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.d_v = config.v_head_dim

        # self.d_qk = config.hidden_size // config.num_attention_heads
        # self.d_v = config.hidden_size // config.num_attention_heads
        self.d_model = config.hidden_size
        self.is_using_mup = config.is_using_mup

        # TODO @thomasw21: refactor so that we store that default in a single place.
        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        self.q_down_proj = nn.Linear(
            self.d_model,
            self.q_lora_rank,
            bias=False,
        )

        self.kv_down_proj = nn.Linear(
            self.d_model,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )

        self.q_up_proj = nn.Linear(
            self.q_lora_rank,
            self.d_qk * config.num_attention_heads,
            bias=False,
        )

        self.kv_up_proj = nn.Linear(
            self.kv_lora_rank,
            config.num_key_value_heads * (self.d_v + self.qk_nope_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            config.num_attention_heads * self.d_v,
            self.d_model,
            bias=False,
        )

        self.attention = CoreAttention(
            config,
            parallel_config=parallel_config,
            layer_idx=layer_idx,
        )

        self.prefill_kv_len = (
            config.seq_len
        )  # TODO @nouamane: compute based on free memory, because in rope we can surpass seq_len

    def forward(
        self,
        hidden_states,  # [batch_size, seq_length, hidden_size]
        sequence_mask,  # [batch_size, seq_length]
        freqs_cis
    ):

        batch_size, seq_length, _ = hidden_states.shape
        q_latent = self.q_down_proj(hidden_states)
        q_latent = F.rms_norm(q_latent, (q_latent.size(-1),), eps=1e-5)
        query_states = self.q_up_proj(q_latent)

        query_states = (
            query_states.view(batch_size, seq_length, self.n_local_q_heads, self.d_qk)
        )
        q_nope, q_rope = torch.split(query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_rope = apply_rotary_emb(q_rope, freqs_cis)
        query_states = torch.cat([q_rope, q_nope], dim=-1).contiguous()

        kv_latent_plus_rope = self.kv_down_proj(hidden_states)
        kv_latent, k_rope = torch.split(kv_latent_plus_rope, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_rope = (
            k_rope.view(batch_size, seq_length, 1, self.qk_rope_head_dim)
        )
        k_rope = apply_rotary_emb(k_rope, freqs_cis)
        k_rope = k_rope.expand(-1, -1, self.n_local_kv_heads, -1) # [batch_size, seq_length, n_local_kv_heads, qk_rope_head_dim]

        kv_latent = F.rms_norm(kv_latent, (kv_latent.size(-1),), eps=1e-5)

        kv = self.kv_up_proj(kv_latent)
        k_nope, value_states = torch.split(kv, [self.n_local_kv_heads * self.qk_nope_head_dim, self.n_local_kv_heads * self.d_v], dim=-1)

        k_nope = (
            k_nope.view(batch_size, seq_length, self.n_local_kv_heads, self.qk_nope_head_dim)
        )
        key_states = torch.cat([k_rope, k_nope], dim=-1).contiguous()

        value_states = (
            value_states.view(batch_size, seq_length, self.n_local_kv_heads, self.d_v)
        )
        q_sequence_mask = sequence_mask
        kv_sequence_mask = sequence_mask
        kv_length = key_states.shape[1]
        # [batch_size, seq_length, num_heads, d_qk]
        # Shaping for use in `flash-attn` version of flash-attn: `flash_attn_unpadded_func`
        query_states = query_states.view(
            batch_size * seq_length, self.n_local_q_heads, self.d_qk
        )  # [batch_size * q_length, self.n_heads, d_qk]

        key_states = key_states.view(
            batch_size * kv_length, self.n_local_kv_heads, self.d_qk
        )  # [batch_size * kv_length, self.n_heads, d_qk]
        value_states = value_states.view(
            batch_size * kv_length, self.n_local_kv_heads, self.d_v
        )  # [batch_size * kv_length, self.n_heads, d_v]

        attention_output = self.attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            q_sequence_mask=q_sequence_mask,
            kv_sequence_mask=kv_sequence_mask,
        )

        attention_output = (
            attention_output.contiguous().view(batch_size, seq_length, self.n_local_q_heads * self.d_v) # no transpose, we are using batch_size, seq_length for first two dims
        )
        output = self.o_proj(attention_output)

        return {"hidden_states": output, "sequence_mask": sequence_mask}

class GLUActivation(nn.Module):
    def __init__(self, act_fn_name: str):
        super().__init__()
        self.act = ACT2FN[act_fn_name]

    def forward(self, merged_states: torch.Tensor):
        gate_states, up_states = torch.split(merged_states, merged_states.shape[-1] // 2, dim=-1)
        return self.act(gate_states) * up_states

class MLP(nn.Module):
    def __init__(
        self,
        config: DeepSeekV3Config,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        is_expert: bool = False,
    ):
        super().__init__()

        # For experts, use regular nn.Linear instead of tensor parallel layers
        if is_expert:
            # Expert implementation without tensor parallelism
            self.gate_up_proj = nn.Linear(
                config.hidden_size,
                2 * config.intermediate_size,
                bias=False,
            )
            self.down_proj = nn.Linear(
                config.intermediate_size,
                config.hidden_size,
                bias=False,
            )
        else:
            # Regular MLP with tensor parallelism
            tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
            tp_linear_async_communication = (
                parallel_config.tp_linear_async_communication if parallel_config is not None else False
            )

            gate_up_contiguous_chunks = (
                config.intermediate_size,  # shape of gate_linear
                config.intermediate_size,  # shape of up_linear
            )
            self.gate_up_proj = TensorParallelColumnLinear(
                config.hidden_size,
                2 * config.intermediate_size,
                pg=tp_pg,
                mode=tp_mode,
                bias=False,
                async_communication=tp_linear_async_communication,
                contiguous_chunks=gate_up_contiguous_chunks,
                tp_recompute_allgather=parallel_config.tp_recompute_allgather,
            )
            self.down_proj = TensorParallelRowLinear(
                config.intermediate_size,
                config.hidden_size,
                pg=tp_pg,
                mode=tp_mode,
                bias=False,
                async_communication=tp_linear_async_communication and tp_mode is TensorParallelLinearMode.REDUCE_SCATTER,
            )

        self.split_silu_mul = GLUActivation(config.hidden_act)
        self.is_expert = is_expert

    def forward(self, hidden_states):  # [batch_size, seq_length, hidden_dim]
        merged_states = self.gate_up_proj(hidden_states)
        hidden_states = self.down_proj(self.split_silu_mul(merged_states))

        # Experts might need to return just the tensor rather than a dict
        # depending on how your MoE implementation expects it
        if self.is_expert:
            return hidden_states
        else:
            return {"hidden_states": hidden_states}

class DeepSeekV3Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """
    def __init__(self, config: DeepSeekV3Config):
        """
        Initializes the Gate module.

        Args:
            config (DeepSeekV3Config): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = config.hidden_size
        self.topk = config.n_activated_experts
        self.n_groups = config.n_expert_groups
        self.topk_groups = config.n_limited_groups
        self.score_func = config.score_func
        self.route_scale = config.route_scale
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        self.bias = nn.Parameter(torch.empty(config.n_routed_experts), requires_grad=False) # for aux loss free load balancing

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        scores = F.linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores.clone()
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices # shapes are [batch_size, topk]

class DeepSeekV3MoE(nn.Module):
    def __init__(
        self,
        config: DeepSeekV3Config,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.world_size = tp_pg.size()
        self.rank = tp_pg.rank()

        assert config.n_routed_experts % self.world_size == 0, \
            f"Number of routed experts ({config.n_routed_experts}) must be divisible by world size ({self.world_size})"
        assert config.n_shared_experts % self.world_size == 0, \
            f"Number of shared experts ({config.n_shared_experts}) must be divisible by world size ({self.world_size})"

        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.n_local_routed_experts = self.n_routed_experts // self.world_size
        self.n_local_shared_experts = self.n_shared_experts // self.world_size
        self.n_activated_experts = config.n_activated_experts

        # Expert index ranges for this worker
        self.routed_start_idx = self.rank * self.n_local_routed_experts
        self.routed_end_idx = self.routed_start_idx + self.n_local_routed_experts
        self.shared_start_idx = self.rank * self.n_local_shared_experts
        self.shared_end_idx = self.shared_start_idx + self.n_local_shared_experts

        self.gate = DeepSeekV3Gate(config)

        # Routed experts (distributed across TP workers)
        self.routed_experts = nn.ModuleList([
            MLP(
                config=config,
                parallel_config=parallel_config,
                tp_pg=tp_pg,
                is_expert=True
            ) if self.routed_start_idx <= i < self.routed_end_idx else None
            for i in range(self.n_routed_experts)
        ])

        # Shared experts (also distributed across TP workers)
        self.shared_experts = nn.ModuleList([
            MLP(
                config=config,
                parallel_config=parallel_config,
                tp_pg=tp_pg,
                is_expert=True
            ) if self.shared_start_idx <= i < self.shared_end_idx else None
            for i in range(self.n_shared_experts)
        ])

        self.tp_pg = tp_pg

    def forward(self, hidden_states):
        x = hidden_states

        original_shape = x.size()
        x = x.view(-1, self.hidden_size)

        # Process routed experts
        weights, indices = self.gate(x)
        routed_output = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        expert_load = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).float()

        for i in range(self.routed_start_idx, self.routed_end_idx):
            if counts[i] == 0:
                continue
            expert = self.routed_experts[i]
            idx, top = torch.where(indices == i)
            local_output = expert(x[idx])
            routed_output[idx] += local_output * weights[idx, top, None]

        # Process shared experts - all tokens through the local shared experts
        shared_output = torch.zeros_like(x)
        for i in range(self.shared_start_idx, self.shared_end_idx):
            expert = self.shared_experts[i]
            shared_output += expert(x)

        # All-reduce both outputs to combine across workers
        if self.world_size > 1:
            dist.all_reduce(routed_output, group=self.tp_pg)
            dist.all_reduce(shared_output, group=self.tp_pg)

        # For shared experts, average by total number of experts
        if self.n_shared_experts > 0:
            shared_output = shared_output / self.n_shared_experts

        # Combine outputs
        combined_output = (routed_output + shared_output).view(original_shape)

        return {
            "hidden_states": combined_output,
            "expert_load": expert_load,
        }

class DeepSeekV3Layer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV3Config,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_idx: int,
    ):
        super().__init__()
        self.input_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = DeepSeekV3MLA(
            config=config,
            parallel_config=parallel_config,
            tp_pg=tp_pg,
            layer_idx=layer_idx,
        )
        self.post_attention_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("freqs_cis", precompute_freqs_cis(config=config), persistent=False)
        # Use MoE for layers beyond the dense layers
        if layer_idx >= config.n_dense_layers:
            self.mlp = DeepSeekV3MoE(config=config, parallel_config=parallel_config, tp_pg=tp_pg)
        else:
            self.mlp = MLP(config=config, parallel_config=parallel_config, tp_pg=tp_pg)

        self.recompute_layer = parallel_config.recompute_layer if parallel_config else False
        self.layer_idx = layer_idx

    def _core_forward(
        self,
        hidden_states: Union[torch.Tensor, TensorPointer],
        sequence_mask: Union[torch.Tensor, TensorPointer],
        freqs_cis: Union[torch.Tensor, TensorPointer],
    ) -> List[Union[torch.Tensor, TensorPointer]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        output = self.attn(hidden_states=hidden_states, sequence_mask=sequence_mask, freqs_cis=freqs_cis)
        hidden_states = output["hidden_states"]
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MoE or MLP layers
        mlp_output = self.mlp(hidden_states)
        hidden_states = mlp_output["hidden_states"]
        hidden_states = hidden_states + residual

        expert_load = mlp_output.get("expert_load", None)

        return hidden_states, output["sequence_mask"], expert_load

    def _checkpointed_forward(
        self,
        hidden_states: torch.Tensor,
        sequence_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> List[torch.Tensor]:
        return CheckpointFunction.apply(self._core_forward, True, hidden_states, sequence_mask, freqs_cis)

    def forward(
        self,
        hidden_states: Union[torch.Tensor, TensorPointer],
        sequence_mask: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        # Compute rotary embeddings
        if self.recompute_layer and not isinstance(hidden_states, TensorPointer):
            hidden_states, sequence_mask = self._checkpointed_forward(hidden_states, sequence_mask, self.freqs_cis)
        else:
            hidden_states, sequence_mask, expert_load = self._core_forward(hidden_states, sequence_mask, self.freqs_cis)

        return {
            "hidden_states": hidden_states,
            "sequence_mask": sequence_mask,
            "expert_load": expert_load,
        }

class DeepSeekV3Model(nn.Module):
    """Build pipeline graph for DeepSeek V3"""

    def __init__(
        self,
        config: DeepSeekV3Config,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
    ):
        super().__init__()

        # Set up pipeline blocks for embeddings, layers, and outputs
        self.p2p = P2P(parallel_context.pp_pg, device=torch.device("cuda"))
        self.config = config
        self.parallel_config = parallel_config
        self.parallel_context = parallel_context
        self.tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        # Token embeddings
        self.token_position_embeddings = PipelineBlock(
            p2p=self.p2p,
            module_builder=Embedding,  # Re-use from existing code
            module_kwargs={
                "tp_pg": parallel_context.tp_pg,
                "config": config,
                "parallel_config": parallel_config,
            },
            module_input_keys={"input_ids", "input_mask"},
            module_output_keys={"input_embeds"},
        )

        # Log RoPE configuration
        log_rank(f"Initialize RoPE Theta = {config.rope_theta}", logger=logger, level=logging.INFO, rank=0)
        log_rank(f"YARN Scaling: factor={config.rope_factor}, beta_fast={config.beta_fast}, beta_slow={config.beta_slow}", 
                logger=logger, level=logging.INFO, rank=0)

        # Transformer layers
        self.decoder = nn.ModuleList([
            PipelineBlock(
                p2p=self.p2p,
                module_builder=DeepSeekV3Layer,
                module_kwargs={
                    "config": config,
                    "parallel_config": parallel_config,
                    "tp_pg": parallel_context.tp_pg,
                    "layer_idx": layer_idx,
                },
                module_input_keys={"hidden_states", "sequence_mask"},
                module_output_keys={"hidden_states", "sequence_mask", "expert_load"},
            )
            for layer_idx in range(config.num_hidden_layers)
        ])

        # Final normalization and output layers
        self.final_layer_norm = PipelineBlock(
            p2p=self.p2p,
            module_builder=TritonRMSNorm,
            module_kwargs={"hidden_size": config.hidden_size, "eps": config.rms_norm_eps},
            module_input_keys={"input"},
            module_output_keys={"hidden_states"},
        )

        self.lm_head = PipelineBlock(
            p2p=self.p2p,
            # Returns sharded logits that will need to be gathered
            module_builder=TensorParallelColumnLinear,
            module_kwargs={
                "in_features": config.hidden_size,
                "out_features": config.vocab_size,
                "pg": parallel_context.tp_pg,
                "bias": False,
                "mode": self.tp_mode,
                "async_communication": tp_linear_async_communication,
                "tp_recompute_allgather": parallel_config.tp_recompute_allgather if parallel_config else False,
            },
            module_input_keys={"x"},
            module_output_keys={"logits"},
        )

        self.cast_to_fp32 = PipelineBlock(
            p2p=self.p2p,
            module_builder=lambda: lambda x: x.float(),
            module_kwargs={},
            module_input_keys={"x"},
            module_output_keys={"output"},
        )

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
        input_mask: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
    ):
        return self.forward_with_hidden_states(input_ids=input_ids, input_mask=input_mask)[0]

    def forward_with_hidden_states(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
        input_mask: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
    ):
        # All tensors are optional as most ranks don't need anything from the dataloader
        output = self.token_position_embeddings(input_ids=input_ids, input_mask=input_mask)

        hidden_encoder_states = {
            "hidden_states": output["input_embeds"],
            "sequence_mask": input_mask,
        }

        all_expert_loads = []

        for encoder_block in self.decoder:
            block_output = encoder_block(**hidden_encoder_states)
            hidden_encoder_states = {
                "hidden_states": block_output["hidden_states"],
                "sequence_mask": block_output["sequence_mask"],
            }
            all_expert_loads.append(block_output.get("expert_load", None))

        hidden_states = self.final_layer_norm(input=hidden_encoder_states["hidden_states"])["hidden_states"]
        sharded_logits = self.lm_head(x=hidden_states)["logits"]
        fp32_sharded_logits = self.cast_to_fp32(x=sharded_logits)["output"]

        return fp32_sharded_logits, hidden_states, all_expert_loads

    def get_block_compute_costs(self):
        """Computes the compute cost of each block in the model so that we can do a better job of load balancing."""
        model_config = self.config
        block_compute_costs = {
            # CausalSelfAttention (qkv proj + attn out) + MLP
            DeepSeekV3Layer: model_config.hidden_size * model_config.q_lora_rank + \
                model_config.hidden_size * (model_config.kv_lora_rank + model_config.qk_rope_head_dim) + \
                model_config.q_lora_rank * model_config.num_attention_heads * (model_config.qk_nope_head_dim + model_config.qk_rope_head_dim) + \
                model_config.kv_lora_rank * model_config.num_attention_heads * (model_config.qk_nope_head_dim + model_config.v_head_dim) + \
                model_config.hidden_size * (model_config.v_head_dim * model_config.num_attention_heads) + \
                3 * model_config.intermediate_size * model_config.hidden_size, #TODO: find a way to distinguish between MoE and dense here
            # This is the last lm_head
            TensorParallelColumnLinear: model_config.vocab_size * model_config.hidden_size,
        }
        return block_compute_costs

class Loss(nn.Module):
    def __init__(self, tp_pg: dist.ProcessGroup):
        super().__init__()
        self.tp_pg = tp_pg

    def forward(
        self,
        sharded_logits: torch.Tensor,  # [batch_size, seq_length, logits]
        label_ids: torch.Tensor,  # [batch_size, seq_length]
        label_mask: torch.Tensor,  # [batch_size, seq_length]
    ) -> Dict[str, torch.Tensor]:
        # Calculate cross entropy loss
        ce_loss = sharded_cross_entropy(
            sharded_logits, label_ids, group=self.tp_pg, dtype=torch.float
        )
        loss = masked_mean(ce_loss, label_mask, dtype=torch.float)

        return {"loss": loss}

@torch.jit.script
def masked_mean(loss, label_mask, dtype):
    # type: (Tensor, Tensor, torch.dtype) -> Tensor
    return (loss * label_mask).sum(dtype=dtype) / label_mask.sum()

class DeepSeekV3ForTraining(NanotronModel):
    def __init__(
        self,
        config: DeepSeekV3Config,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
        random_states: Optional[RandomStates] = None,
    ):
        super().__init__()
        self.model = DeepSeekV3Model(config=config, parallel_context=parallel_context, parallel_config=parallel_config)

        # Set up loss function
        self.loss = PipelineBlock(
            p2p=self.model.p2p,
            module_builder=Loss,
            module_kwargs={"tp_pg": parallel_context.tp_pg},
            module_input_keys={
                "sharded_logits",
                "label_ids",
                "label_mask",
            },
            module_output_keys={"loss"},
        )

        self.parallel_context = parallel_context
        self.config = config
        self.parallel_config = parallel_config

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],
        input_mask: Union[torch.Tensor, TensorPointer],
        label_ids: Union[torch.Tensor, TensorPointer],
        label_mask: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        # Forward pass
        sharded_logits, _, all_expert_loads = self.model.forward_with_hidden_states(
            input_ids=input_ids, input_mask=input_mask
        )

        loss = self.loss(
            sharded_logits=sharded_logits,
            label_ids=label_ids,
            label_mask=label_mask,
        )["loss"]

        self.adjust_routing_biases(all_expert_loads, self.config.gamma)

        return {"loss": loss}

    @torch.no_grad()
    def adjust_routing_biases(self, all_expert_loads, gamma=0.01):
        """Adjust the routing biases for load balancing."""

        for layer, expert_load in zip(self.model.decoder, all_expert_loads):
            if expert_load is None:
                continue
            moe_layer = layer.pp_block.mlp
            if isinstance(moe_layer, DeepSeekV3MoE):
                # Gather expert load across TP workers
                dist.all_reduce(expert_load, group=self.parallel_context.tp_pg)

                avg_load = expert_load.mean()

                overloaded = expert_load > avg_load
                underloaded = expert_load < avg_load

                # Adjust biases
                moe_layer.gate.routing_bias[overloaded] -= gamma
                moe_layer.gate.routing_bias[underloaded] += gamma

    def get_block_compute_costs(self):
        """Return the compute costs of each block in the model."""
        return self.model.get_block_compute_costs()

    def get_embeddings_lm_head_tied_names(self):
        """Get the names of the tied embeddings and lm_head weights"""
        if self.config.tie_word_embeddings is True:
            return ["model.token_position_embeddings.pp_block.token_embedding.weight", "model.lm_head.pp_block.weight"]
        else:
            return []

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        """Get flops per second for a given model"""
        world_size = self.parallel_context.world_pg.size()
        try:
            num_key_values_heads = self.config.num_key_value_heads
        except AttributeError:
            num_key_values_heads = self.config.num_attention_heads

        model_flops, hardware_flops = get_flops(
            num_layers=self.config.num_hidden_layers,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_key_value_heads=num_key_values_heads,
            vocab_size=self.config.vocab_size,
            ffn_hidden_size=self.config.intermediate_size,
            seq_len=sequence_length,
            batch_size=global_batch_size,
        )

        model_flops_per_s = model_flops / (iteration_time_in_sec * world_size * 1e12)
        hardware_flops_per_s = hardware_flops / (iteration_time_in_sec * world_size * 1e12)
        return model_flops_per_s, hardware_flops_per_s

    @torch.no_grad()
    def init_model_randomly(self, config: Config):
        """Initialize model parameters randomly.
        Note:
            Layernorm weight all 0 or 1 depending on `apply_layernorm_1p`
        """
        init_method = config.model.init_method
        if isinstance(init_method, RandomInit):
            parametrizator_cls = StandardParametrizator
        elif isinstance(init_method, SpectralMupInit):
            parametrizator_cls = SpectralMupParametrizator
        else:
            raise ValueError(f"Unknown init method {init_method}")

        parametrizator = parametrizator_cls(config=config.model)

        log_rank(
            f"Parametrizing model parameters using {parametrizator.__class__.__name__}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        model = self
        initialized_parameters = set()
        # Handle tensor parallelism
        module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
        # Fix the root_model
        module_id_to_prefix[id(model)] = ""

        for param_name, param in model.named_parameters():
            assert isinstance(param, NanotronParameter)

            module_name, param_name = param_name.rsplit(".", 1)

            if param.is_tied:
                tied_info = param.get_tied_info()
                full_param_name = tied_info.get_full_name_from_module_id_to_prefix(
                    module_id_to_prefix=module_id_to_prefix
                )
            else:
                full_param_name = f"{module_name}.{param_name}"

            if full_param_name in initialized_parameters:
                # Already initialized
                continue

            module = model.get_submodule(module_name)
            parametrizator.parametrize(param_name, module)

            assert full_param_name not in initialized_parameters
            initialized_parameters.add(full_param_name)

        assert initialized_parameters == {
            param.get_tied_info().get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)
            if param.is_tied
            else name
            for name, param in model.named_parameters()
        }, f"Somehow the initialized set of parameters don't match:\n - Expected: { {name for name, _ in model.named_parameters()} }\n - Got: {initialized_parameters}"

# Note: We reuse the get_flops function from the llama model
# TODO: fix this to be correct for DeepSeekV3
def get_flops(
    num_layers,
    hidden_size,
    num_heads,
    num_key_value_heads,
    vocab_size,
    seq_len,
    ffn_hidden_size,
    batch_size=1,
):
    if num_key_value_heads is None:
        num_key_value_heads = num_heads
    hidden_size_per_head = hidden_size // num_heads
    # In the following we mark the reduced dimension with parentheses
    # decoder
    # self attention
    ## qkv projection
    decoder_qkv_proj_flops_fwd = (
        2 * num_layers * batch_size * seq_len * (hidden_size) * num_heads * hidden_size_per_head
        + 2 * num_layers * batch_size * seq_len * (hidden_size) * 2 * num_key_value_heads * hidden_size_per_head
    )
    ## qk logits
    decoder_qk_logits_flops_fwd = 2 * num_layers * batch_size * num_heads * seq_len * (hidden_size_per_head) * seq_len
    ## v logits
    decoder_v_logits_flops_fwd = 2 * num_layers * batch_size * num_heads * seq_len * (seq_len) * hidden_size_per_head
    ## attn out
    decoder_attn_out_flops_fwd = (
        2 * num_layers * batch_size * num_heads * seq_len * (hidden_size_per_head) * hidden_size
    )
    # FF
    ## 1st layer
    decoder_ffn_1_flops_fwd = 4 * num_layers * batch_size * seq_len * (hidden_size) * ffn_hidden_size
    ## 2nd layer
    decoder_ffn_2_flops_fwd = 2 * num_layers * batch_size * seq_len * (ffn_hidden_size) * hidden_size

    decoder_flops_fwd = (
        decoder_qkv_proj_flops_fwd
        + decoder_qk_logits_flops_fwd
        + decoder_v_logits_flops_fwd
        + decoder_attn_out_flops_fwd
        + decoder_ffn_1_flops_fwd
        + decoder_ffn_2_flops_fwd
    )

    # lm head
    lm_head_flops_fwd = 2 * batch_size * seq_len * (hidden_size) * vocab_size

    # the bwd pass requires double the flops in case of matmuls to calculate the gradients with respect to
    # both input and weight tensors
    model_flops = 3 * (decoder_flops_fwd + lm_head_flops_fwd)  # 1 for fwd + 2 for bwd

    hardware_flops = model_flops  # TODO: This is a placeholder for now

    return model_flops, hardware_flops
