"""Speculative decoding settings: method, window limits, controller and lookup parameters.

``SpeculativeConfig`` is a frozen host value and imports no torch. By default speculation is
disabled (``enabled=False``) until measurements on the target hardware pass the pre-written gates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

from ayaka.configs.base import ConfigMixin
from ayaka.core.errors import ConfigError
from ayaka.utils.validation import require_bool, require_int, require_number

__all__ = [
    "SPECULATIVE_CONTROLLERS",
    "SPECULATIVE_METHODS",
    "SpeculativeConfig",
    "SpeculativeController",
    "SpeculativeMethod",
]

type SpeculativeMethod = Literal["lookup", "mtp", "token_recycling"]
type SpeculativeController = Literal["cost", "fixed"]

SPECULATIVE_METHODS: Final[tuple[SpeculativeMethod, ...]] = (
    "lookup",
    "mtp",
    "token_recycling",
)
SPECULATIVE_CONTROLLERS: Final[tuple[SpeculativeController, ...]] = ("cost", "fixed")


@dataclass(frozen=True, slots=True)
class SpeculativeConfig(ConfigMixin):
    """Settings for lossless speculative decoding.

    Attributes:
        enabled: Master switch. Off by default until measured gates pass.
        method: Drafter kind: ``"lookup"``, ``"mtp"``, or ``"token_recycling"``.
        max_draft_tokens: Maximum draft tokens ``k`` proposed per request per round.
        controller: ``"cost"`` (cost-aware adaptive choice) or ``"fixed"`` (always ``fixed_k``).
        fixed_k: Draft window length used by the ``"fixed"`` controller.
        min_gain: Minimum relative throughput improvement over ``k=0`` for the cost controller to
            enable drafting.
        ema: Smoothing factor in ``(0, 1]`` for acceptance and step-time EMAs.
        lookup_min_match: Minimum suffix match length for n-gram lookup to propose tokens.
        lookup_max_match: Maximum suffix match length considered by n-gram lookup.
        lookup_candidates: Maximum recent candidate end positions inspected per trigram lookup.
        accept_prediction: Whether OpenAI ``prediction`` hints feed a per-request read-only index.
        mtp_depth: Number of sequential MTP draft steps per round (1 default; 2 experimental).
        mtp_vocab_limit: Top-frequent vocabulary rows used by an experimental reduced draft head
            (0 uses the full target head).
        lookup_override_match: When positive and running a model drafter (MTP), prefer
            the lookup proposal when its match length reaches this threshold.
    """

    enabled: bool = False
    method: SpeculativeMethod = "lookup"
    max_draft_tokens: int = 8
    controller: SpeculativeController = "cost"
    fixed_k: int = 2
    min_gain: float = 0.05
    ema: float = 0.05
    lookup_min_match: int = 3
    lookup_max_match: int = 32
    lookup_candidates: int = 4
    accept_prediction: bool = True
    mtp_depth: int = 1
    mtp_vocab_limit: int = 0
    lookup_override_match: int = 0

    def __post_init__(self) -> None:
        require_bool(self.enabled, "speculative.enabled")
        if self.method not in SPECULATIVE_METHODS:
            raise ConfigError(
                "speculative.method",
                "INVALID_CHOICE",
                f"must be one of {list(SPECULATIVE_METHODS)}, got {self.method!r}",
            )
        if self.controller not in SPECULATIVE_CONTROLLERS:
            raise ConfigError(
                "speculative.controller",
                "INVALID_CHOICE",
                f"must be one of {list(SPECULATIVE_CONTROLLERS)}, got {self.controller!r}",
            )
        require_int(self.max_draft_tokens, "speculative.max_draft_tokens", minimum=1)
        require_int(self.fixed_k, "speculative.fixed_k", minimum=0)
        if self.fixed_k > self.max_draft_tokens:
            raise ConfigError(
                "speculative.fixed_k",
                "INVALID_RANGE",
                f"fixed_k ({self.fixed_k}) exceeds max_draft_tokens ({self.max_draft_tokens})",
            )
        min_gain = require_number(self.min_gain, "speculative.min_gain", minimum=0.0)
        if not math.isfinite(min_gain):
            raise ConfigError("speculative.min_gain", "INVALID_RANGE", "must be finite")
        object.__setattr__(self, "min_gain", float(min_gain))

        ema = require_number(self.ema, "speculative.ema", minimum=0.0)
        if not math.isfinite(ema) or ema <= 0.0 or ema > 1.0:
            raise ConfigError("speculative.ema", "INVALID_RANGE", f"must be in (0, 1], got {ema}")
        object.__setattr__(self, "ema", float(ema))

        require_int(self.lookup_min_match, "speculative.lookup_min_match", minimum=1)
        require_int(
            self.lookup_max_match,
            "speculative.lookup_max_match",
            minimum=self.lookup_min_match,
        )
        require_int(self.lookup_candidates, "speculative.lookup_candidates", minimum=1)
        require_bool(self.accept_prediction, "speculative.accept_prediction")
        require_int(self.mtp_depth, "speculative.mtp_depth", minimum=1)
        require_int(self.mtp_vocab_limit, "speculative.mtp_vocab_limit", minimum=0)
        require_int(self.lookup_override_match, "speculative.lookup_override_match", minimum=0)
