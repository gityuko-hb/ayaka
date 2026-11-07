"""Where a model checkpoint comes from and how much it is trusted.

``ModelSourceConfig`` is a frozen host value. It only validates syntax at
construction; it never touches the network or the filesystem. Resolving the
checkpoint is the loader's job.

::

    RequestFormat ----+
      (weight format) |
                      v
                ModelSourceConfig --> tokenizer_config_from_source()
                      ^                 (see ayaka.configs.tokenizer)
                      |
    ModelTrustPolicy -+
      (run repo code?)

Serialization (``to_dict``, ``canonical_json``, ``fingerprint``) comes from
``ConfigMixin``; enums serialize to their string values.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from typing import cast

from ayaka.configs.base import ConfigMixin
from ayaka.utils.validation import require_bool, require_text

_HUB_ID = re.compile(r"[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*")
_COMMIT_SHA = re.compile(r"[0-9a-fA-F]{40}")


class RequestFormat(enum.StrEnum):
    """Checkpoint serialization format requested by the caller.

    Members are strings (``"auto"``, ``"safetensors"``, ``"pt"``, ``"dummy"``) so
    they serialize stably, but :class:`ModelSourceConfig` only accepts the enum
    members themselves, not raw strings.
    """

    AUTO = "auto"
    """Auto-detect format: inspect available checkpoint files and pick the best match."""

    SAFETENSORS = "safetensors"
    """Require SafeTensors format (single-file or sharded); reject legacy formats."""

    PT = "pt"
    """Legacy PyTorch pickle-based weights (*.bin / *.pt).

    Pickle can execute arbitrary code on load; prefer :attr:`SAFETENSORS`.
    """

    DUMMY = "dummy"
    """Random weights of the right shape; used for shape, memory, and plumbing tests."""


class ModelTrustPolicy(enum.StrEnum):
    """Whether code shipped inside a checkpoint repository may be executed.

    Defaults to :attr:`STRICT`; opting in to :attr:`ALLOW_REMOTE_CODE` is an explicit
    decision that should only be made for repositories you trust.
    """

    STRICT = "strict"
    """Strict security: refuse execution of any arbitrary code from the model checkpoint."""

    ALLOW_REMOTE_CODE = "allow_remote_code"
    """Allow execution of Python code bundled within the model checkpoint repository."""


@dataclass(frozen=True, slots=True)
class ModelSourceConfig(ConfigMixin):
    """Immutable configuration specifying model checkpoint origins and security policies.

    Identifies whether the model resides on the local filesystem or the HuggingFace
    Hub, which revision to pin, whether offline resolution is enforced, and
    whether remote custom code execution is permitted.

    Attributes:
        model: Local directory path (e.g. ``"/models/llama-3-8b"``) or HuggingFace
            Hub repo identifier (e.g. ``"meta-llama/Llama-3-8B"``).
        revision: Specific Git commit SHA, branch name, or tag on the HuggingFace
            Hub. An empty string (``""``) uses the Hub's default revision.
        tokenizer: Optional hub repo ID or directory path containing tokenizer
            files. When empty (default), falls back to :attr:`model`.
        format: Desired checkpoint serialization format. See :class:`RequestFormat`.
            Default is :attr:`RequestFormat.AUTO`.
        trust: Remote code execution security policy. See :class:`ModelTrustPolicy`.
            Default is :attr:`ModelTrustPolicy.STRICT`.
        offline: When ``True``, never touches the network and resolves exclusively
            from the local hub cache directory.
        tokenizer_revision: Revision used only when :attr:`tokenizer` names a
            different repo than :attr:`model`; ignored otherwise. Same forms as
            :attr:`revision`. :attr:`revision` is never applied to a different
            tokenizer repo, because a commit of one repo does not exist in another.
        reproducible: When ``True``, requires :attr:`revision` to be a full
            40-character commit SHA (and :attr:`tokenizer_revision` too, when the
            tokenizer is a different repo) and rejects branch names, tags and the
            empty default. The check is purely syntactic and runs at construction, before
            the filesystem is consulted, so it also applies when :attr:`model` is a
            local directory (where the SHA is then only a recorded identity).
        cache_dir: Override path for HuggingFace Hub downloads. An empty string
            defaults to ``HF_HOME`` or the standard library cache.

    Raises:
        ValueError: If ``model`` is empty, a text field is not a string, or
            ``reproducible`` is set without a full commit SHA.
        TypeError: If ``format`` or ``trust`` is not an enum member (raw strings
            are rejected, not coerced), or ``offline``/``reproducible`` is not a
            ``bool`` (``TypeError`` from ``require_bool``).

    Example:
        >>> cfg = ModelSourceConfig("org/model", revision="a" * 40, reproducible=True)
        >>> cfg.tokenizer_path
        'org/model'
    """

    model: str
    revision: str = ""
    tokenizer: str = ""
    tokenizer_revision: str = ""
    format: RequestFormat = RequestFormat.AUTO
    trust: ModelTrustPolicy = ModelTrustPolicy.STRICT
    offline: bool = False
    # A branch resolves to a different commit next week, and "the model changed
    # under us" is costly to debug because nothing recorded that it could.
    reproducible: bool = False
    cache_dir: str = ""

    def __post_init__(self) -> None:
        require_text(self.model, "model")
        for name in ("revision", "tokenizer", "tokenizer_revision", "cache_dir"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")
        if not isinstance(cast("object", self.format), RequestFormat):
            raise TypeError("format must be a RequestFormat member")
        if not isinstance(cast("object", self.trust), ModelTrustPolicy):
            raise TypeError("trust must be a ModelTrustPolicy member")
        require_bool(self.offline, "offline")
        require_bool(self.reproducible, "reproducible")
        if self.reproducible and _COMMIT_SHA.fullmatch(self.revision) is None:
            raise ValueError(
                "reproducible=True requires a full 40-character commit SHA; "
                "branches and tags can move"
            )
        if (
            self.reproducible
            and self.has_separate_tokenizer
            and _COMMIT_SHA.fullmatch(self.tokenizer_revision) is None
        ):
            raise ValueError(
                "reproducible=True requires tokenizer_revision to be a full "
                "40-character commit SHA when the tokenizer is a different repo"
            )

    @property
    def tokenizer_path(self) -> str:
        """Effective path or repo ID for the tokenizer, falling back to ``model``."""
        return self.tokenizer or self.model

    @property
    def has_separate_tokenizer(self) -> bool:
        """Whether ``tokenizer`` is set and names something other than ``model``."""
        return bool(self.tokenizer) and self.tokenizer != self.model

    @property
    def effective_tokenizer_revision(self) -> str:
        """Revision to load the tokenizer at; ``""`` means the Hub default.

        ``tokenizer_revision`` for a separate tokenizer repo, otherwise ``revision``.
        """
        return self.tokenizer_revision if self.has_separate_tokenizer else self.revision

    @property
    def is_dummy(self) -> bool:
        """Whether this configuration requests synthetic dummy weights for testing."""
        return self.format is RequestFormat.DUMMY

    @property
    def allows_remote_code(self) -> bool:
        """Whether executing checkpoint-supplied custom Python code is allowed."""
        return self.trust is ModelTrustPolicy.ALLOW_REMOTE_CODE

    def looks_like_hub_id(self) -> bool:
        """Whether ``model`` has the ``org/name`` shape of a hub repo id.

        Exactly one ``/`` with a non-empty owner and name on each side, each
        starting with an alphanumeric and made of letters, digits, ``_``, ``-``
        and ``.``. Relative (``./x``, ``../x``), absolute, home-relative, drive
        and backslash paths therefore never match.

        Heuristic on purpose, and only a hint: the resolver checks the local
        filesystem first regardless, because a local directory named ``org/name``
        must win over a hub lookup or a typo silently downloads a stranger's
        model.

        Returns:
            ``True`` if ``model`` conforms to the ``org/name`` hub naming pattern.
        """
        return _HUB_ID.fullmatch(self.model) is not None
