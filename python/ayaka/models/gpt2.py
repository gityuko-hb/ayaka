"""GPT-2: learned position embeddings, pre-norm blocks, GELU-tanh feed-forward, tied output head.

The block is the original one::

    x = x + attn(ln_1(x))        # multi-head causal attention with biases
    x = x + mlp(ln_2(x))         # up, GELU (tanh form), down
    ...
    hidden = ln_f(x)             # logits = hidden @ wte.T

The architecture is the one of ``GPT2LMHeadModel``; the knobs of its Hugging Face
config that change the arithmetic (``scale_attn_by_inverse_layer_idx``,
``reorder_and_upcast_attn``) are not supported and are not read.

Checkpoint layout: Hugging Face stores the four projections of a block as
``Conv1D`` weights, ``[in, out]``, and the attention projection already fused
(``c_attn`` is ``[hidden, 3 * hidden]``, columns ordered query, key, value). A
:class:`~ayaka.layers.linear.Linear` is ``[out, in]``, so those four weights are
transposed when they are loaded and nothing is fused. The checkpoint may be the
bare ``GPT2Model`` (names without ``transformer.``) or ``GPT2LMHeadModel`` (with it).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from torch import nn

from ayaka.core.errors import ModelLoadError
from ayaka.layers.attention.dense import Attention
from ayaka.layers.embedding.token import LMHead, TokenEmbedding
from ayaka.layers.ffn.dense import MLP
from ayaka.layers.norm import LayerNorm
from ayaka.models.base import CausalLM
from ayaka.weights.mapping import WeightMapping

if TYPE_CHECKING:
    import torch

    from ayaka.configs.model import ModelConfig
    from ayaka.layers.context import ForwardContext
    from ayaka.ops.backend import OpsBackend
    from ayaka.utils.dtypes import DType

__all__ = ["GPT2LMHeadModel"]

#: ``hidden_act`` values of GPT-2 configs and whether each is the tanh form of GELU.
_ACTIVATIONS = {"gelu_new": True, "gelu_pytorch_tanh": True, "gelu": False}

# ``(?:transformer\.)?`` because the base checkpoint (``GPT2Model``) has no prefix
# and the one with the head (``GPT2LMHeadModel``) has it.
_PREFIX = r"(?:transformer\.)?"
_RENAMES = (
    (rf"{_PREFIX}h\.(\d+)\.(ln_1|ln_2)\.(weight|bias)", r"layers.\1.\2.\3"),
    (rf"{_PREFIX}h\.(\d+)\.attn\.c_attn\.(weight|bias)", r"layers.\1.attn.qkv_proj.\2"),
    (rf"{_PREFIX}h\.(\d+)\.attn\.c_proj\.(weight|bias)", r"layers.\1.attn.o_proj.\2"),
    (rf"{_PREFIX}h\.(\d+)\.mlp\.c_fc\.(weight|bias)", r"layers.\1.mlp.up_proj.\2"),
    (rf"{_PREFIX}h\.(\d+)\.mlp\.c_proj\.(weight|bias)", r"layers.\1.mlp.down_proj.\2"),
    (rf"{_PREFIX}(wte|wpe|ln_f)\.(weight|bias)", r"\1.\2"),
)
#: Old checkpoints store the causal mask as a tensor, per layer; ours is computed.
_MASKS = (rf"{_PREFIX}h\.\d+\.attn\.(?:bias|masked_bias)",)


class _Block(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.ln_1 = LayerNorm(hidden, eps=config.norm_eps, ops=ops, dtype=dtype, device=device)
        self.attn = Attention(
            hidden,
            config.num_attention_heads,
            config.num_kv_heads,
            config.head_dim,
            layer_id=layer_id,
            bias=True,
            ops=ops,
            dtype=dtype,
            device=device,
        )
        self.ln_2 = LayerNorm(hidden, eps=config.norm_eps, ops=ops, dtype=dtype, device=device)
        self.mlp = MLP(
            hidden,
            config.intermediate_size,
            bias=True,
            tanh=_ACTIVATIONS[config.hidden_act],
            ops=ops,
            dtype=dtype,
            device=device,
        )

    def forward(self, x: torch.Tensor, context: ForwardContext) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), context)
        return x + self.mlp(self.ln_2(x))


class GPT2LMHeadModel(CausalLM):
    """GPT-2 with a language-model head.

    Parameters are ``wte`` and ``wpe`` (token and learned position embeddings),
    ``layers.N.{ln_1, attn.qkv_proj, attn.o_proj, ln_2, mlp.up_proj, mlp.down_proj}``,
    ``ln_f`` and ``lm_head``. The head shares ``wte``'s table unless the config
    says ``tie_word_embeddings`` is false.

    Positions come from ``context.positions``, so a step may start anywhere in a
    sequence (chunked prefill, decode). A position at or past
    ``max_position_embeddings`` is not checked here: the op does not check ids,
    and the request is refused when it is admitted.

    Raises:
        ModelLoadError: If the config is not one this implementation runs:
            grouped-query attention, ``num_attention_heads * head_dim`` different
            from ``hidden_size``, or an activation other than ``gelu_new``,
            ``gelu_pytorch_tanh`` and ``gelu``.
    """

    stored_transposed: ClassVar[tuple[str, ...]] = (
        r"layers\.\d+\.attn\.qkv_proj\.weight",
        r"layers\.\d+\.attn\.o_proj\.weight",
        r"layers\.\d+\.mlp\.up_proj\.weight",
        r"layers\.\d+\.mlp\.down_proj\.weight",
    )

    def __init__(
        self,
        config: ModelConfig,
        *,
        ops: OpsBackend,
        dtype: DType,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(config, ops=ops, dtype=dtype, device=device)
        _check_supported(config)
        torch_dtype = dtype.torch_dtype
        hidden = config.hidden_size
        self.wte = TokenEmbedding(
            config.vocab_size, hidden, ops=ops, dtype=torch_dtype, device=device
        )
        self.wpe = TokenEmbedding(
            config.max_position_embeddings, hidden, ops=ops, dtype=torch_dtype, device=device
        )
        self.layers = nn.ModuleList(
            _Block(config, layer_id, ops=ops, dtype=torch_dtype, device=device)
            for layer_id in range(config.num_layers)
        )
        self.ln_f = LayerNorm(
            hidden, eps=config.norm_eps, ops=ops, dtype=torch_dtype, device=device
        )
        self.lm_head = LMHead(hidden, config.vocab_size, ops=ops, dtype=torch_dtype, device=device)
        self.tie_weights()

    @classmethod
    def weight_mapping(cls, config: ModelConfig) -> WeightMapping:
        """Names in a Hugging Face GPT-2 checkpoint (with or without ``transformer.``)."""
        ignore = (*_MASKS, r"lm_head\.weight") if config.tie_word_embeddings else _MASKS
        return WeightMapping(renames=_RENAMES, ignore=ignore)

    def tie_weights(self) -> None:
        """Share ``wte``'s table with ``lm_head`` when the config ties them."""
        if self.config.tie_word_embeddings:
            self.lm_head.tie_weights(self.wte)

    def forward(self, token_ids: torch.Tensor, context: ForwardContext) -> torch.Tensor:
        """Embed the tokens and their positions, run the blocks and normalise.

        Returns ``[N, hidden_size]``; see :meth:`CausalLM.forward`.
        """
        x = self.wte(token_ids) + self.wpe(context.positions)
        for layer in self.layers:
            x = layer(x, context)
        return self.ln_f(x)

    def compute_logits(self, hidden_rows: torch.Tensor) -> torch.Tensor:
        """Project ``[R, hidden_size]`` rows onto the vocabulary with the (tied) head."""
        return self.lm_head(hidden_rows)


def _check_supported(config: ModelConfig) -> None:
    problems: list[str] = []
    if config.num_kv_heads != config.num_attention_heads:
        problems.append(
            f"it uses multi-head attention, but num_kv_heads is {config.num_kv_heads} "
            f"for {config.num_attention_heads} heads"
        )
    if config.num_attention_heads * config.head_dim != config.hidden_size:
        problems.append(
            f"num_attention_heads * head_dim is "
            f"{config.num_attention_heads * config.head_dim}, not hidden_size {config.hidden_size}"
        )
    if config.hidden_act not in _ACTIVATIONS:
        problems.append(f"hidden_act {config.hidden_act!r} is not one of {sorted(_ACTIVATIONS)}")
    if problems:
        raise ModelLoadError(
            f"cannot build {config.architecture} from this config: " + "; ".join(problems),
            context={"architecture": config.architecture},
        )
