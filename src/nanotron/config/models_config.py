from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Literal, Optional, Union


@dataclass
class RandomInit:
    std: float


@dataclass
class SpectralMupInit:
    """This is used to initialize the model with spectral mup. Set it to True to use it."""

    use_mup: bool

    def __post_init__(self):
        assert self.use_mup, "Remove `use_mup` if you don't want to use it"


@dataclass
class ExistingCheckpointInit:
    """This is used to initialize from an already existing model (without optimizer, lr_scheduler...)"""

    path: Path


@dataclass
class VellichorConfig:
    """Configuration for a Vellichor model

    Be careful on having a coherent typing as we use it to reconstruct the model from yaml
    """

    bos_token_id: int = 1
    eos_token_id: int = 2
    hidden_act: str = "silu"
    hidden_size: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 11008
    is_vellichor_config: bool = True  # We use this help differentiate models in yaml/python conversion
    use_qk_norm: bool = True
    max_position_embeddings: int = 2048
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    num_key_value_heads: Optional[int] = None
    pad_token_id: Optional[int] = None
    pretraining_tp: int = 1
    rms_norm_eps: float = 1e-6
    rope_scaling: Optional[dict] = None
    rope_theta: float = 10000.0
    rope_interleaved: bool = (
        False  # The default value has been True, but for loading Llama3 checkpoints you have to set it to False
    )
    tie_word_embeddings: bool = False
    use_cache: bool = True
    vocab_size: int = 32000

    q_lora_rank: int = 256
    kv_lora_rank: int = 256
    qk_nope_head_dim: int = 96
    qk_rope_head_dim: int = 32
    v_head_dim: int = 128
    linear_attn_head_dim: int = 96
    linear_attn_num_heads: int = 6
    full_attention_every_n_layers: int = 1

    def __post_init__(self):
        # NOTE: user don't set self._init_method, ModelArgs will set it
        # then we only pass VellichorConfig around
        self._is_using_mup: bool = False
        # self._init_method: Optional[Union[RandomInit, SpectralMupInit, ExistingCheckpointInit]] = None

        # for backward compatibility
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

    @property
    def is_using_mup(self) -> bool:
        return self._is_using_mup

@dataclass
class LlamaConfig:
    """Configuration for a LLAMA model

    Be careful on having a coherent typing as we use it to reconstruct the model from yaml
    """

    bos_token_id: int = 1
    eos_token_id: int = 2
    hidden_act: str = "silu"
    hidden_size: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 11008
    is_llama_config: bool = True  # We use this help differentiate models in yaml/python conversion
    max_position_embeddings: int = 2048
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    num_key_value_heads: Optional[int] = None
    pad_token_id: Optional[int] = None
    pretraining_tp: int = 1
    rms_norm_eps: float = 1e-6
    rope_scaling: Optional[dict] = None
    rope_theta: float = 10000.0
    rope_interleaved: bool = (
        False  # The default value has been True, but for loading Llama3 checkpoints you have to set it to False
    )
    tie_word_embeddings: bool = False
    use_cache: bool = True
    vocab_size: int = 32000
    use_qk_norm: bool = False

    def __post_init__(self):
        # NOTE: user don't set self._init_method, ModelArgs will set it
        # then we only pass LlamaConfig around
        self._is_using_mup: bool = False
        # self._init_method: Optional[Union[RandomInit, SpectralMupInit, ExistingCheckpointInit]] = None

        # for backward compatibility
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

    @property
    def is_using_mup(self) -> bool:
        return self._is_using_mup


@dataclass
class Starcoder2Config:
    """Configuration for a Starcoder2 model

    Be careful on having a coherent typing as we use it to reconstruct the model from yaml
    """

    activation_function: str = "gelu_pytorch_tanh"
    attention_softmax_in_fp32: bool = True  # TODO: not used
    attn_pdrop: float = 0.1
    bos_token_id: int = 49152  # TODO: not used
    embd_pdrop: float = 0.1
    eos_token_id: int = 49152
    global_attn_layers: List[int] = field(default_factory=list)
    grouped_query: bool = False  # GQA
    hidden_size: int = 2048
    initializer_range: float = 0.02  # TODO: not used
    intermediate_size: Optional[int] = None
    is_starcoder2_config: bool = True  # We use this help differentiate models in yaml/python conversion
    layer_norm_epsilon: float = 1e-05
    max_position_embeddings: int = 4096
    multi_query: bool = False  # MQA
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    num_kv_heads: Optional[int] = None
    resid_pdrop: float = 0.1
    rope_theta: Optional[int] = 10000
    scale_attention_softmax_in_fp32: bool = True
    scale_attn_weights: bool = True
    sliding_window_size: Optional[int] = None
    use_position_embeddings: bool = False  # TODO @nouamane this is not used
    use_rotary_embeddings: bool = True
    vocab_size: int = 49280

    def __post_init__(self):
        if self.global_attn_layers is None:
            self.global_attn_layers = []

        if self.grouped_query:
            assert self.num_kv_heads is not None, "num_kv_heads must be specified for grouped query"
            assert self.multi_query is False, "Cannot use both multi_query and grouped_query"

        if not self.multi_query and not self.grouped_query:
            self.multi_query = True

    @property
    def n_embed(self):
        return self.hidden_size

    @property
    def n_head(self):
        return self.num_attention_heads

    @property
    def n_layer(self):
        return self.num_hidden_layers

    @property
    def n_positions(self):
        return self.max_position_embeddings

    @property
    def n_inner(self):
        return self.intermediate_size

@dataclass
class DeepSeekV3Config:
    """
    Configuration for a DeepSeekV3 model
    """

    # Basic model configuration
    bos_token_id: int = 1
    eos_token_id: int = 2
    hidden_act: str = "silu"
    hidden_size: int = 1024  # Same as dim
    initializer_range: float = 0.02
    intermediate_size: int = 4096  # Same as inter_dim
    is_deepseekv3_config: bool = True  # We use this to differentiate models in yaml/python conversion
    seq_len: int = 2048 # this is max_position_embeddings but I think this is a better name
    original_seq_len: int = 2048
    num_attention_heads: int = 16  # Same as n_heads
    num_hidden_layers: int = 24  # Same as n_layers
    num_key_value_heads: Optional[int] = None
    pad_token_id: Optional[int] = None
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    use_cache: bool = False # for training
    vocab_size: int = 32000

    # DeepSeekV3 specific parameters
    max_batch_size: int = 4 # this should be set to whatever micro-batch size for training
    dtype: str = "bfloat16" # TODO: potentially not needed, but not sure
    n_dense_layers: int = 1 # first few layers are not MoE

    # MoE configuration
    n_routed_experts: int = 6
    n_shared_experts: int = 2
    n_activated_experts: int = 4
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_scale: float = 1.0
    moe_intermediate_size: int = 256
    gamma: float = 0.01

    # MLA configuration
    q_lora_rank: int = 256
    kv_lora_rank: int = 256
    qk_nope_head_dim: int = 96
    qk_rope_head_dim: int = 32
    v_head_dim: int = 128
    attn_impl: Literal["naive", "fused"] = "naive"

    # Positional encoding configuration
    rope_theta: float = 10000.0
    rope_factor: float = 1.0
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0
    rope_scaling: Optional[dict] = None
    rope_interleaved: bool = False

    def __post_init__(self):
        # NOTE: user don't set self._init_method, ModelArgs will set it
        self._is_using_mup: bool = False
        # self._init_method: Optional[Union[RandomInit, SpectralMupInit, ExistingCheckpointInit]] = None

        # for backward compatibility
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

    @property
    def is_using_mup(self) -> bool:
        return self._is_using_mup

NanotronConfigs = Union[VellichorConfig, LlamaConfig, Starcoder2Config, DeepSeekV3Config, Any]
