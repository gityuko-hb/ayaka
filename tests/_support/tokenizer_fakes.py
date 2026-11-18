"""A byte-level fake tokenizer, so the whole tokenizer layer is testable without
``transformers``, ``tokenizers`` or ``tiktoken``.

Lives in ``tests/_support`` (on ``pythonpath``) so test modules can import it.

Ids 0-255 are single bytes; 256-259 are the special tokens below. ``encode`` never
parses special tokens out of text (like the tiktoken adapter); the subclass
:class:`RenderedFake` adds ``encode_rendered_batch`` that does.
"""

from __future__ import annotations

import codecs
import json
from typing import TYPE_CHECKING, Any

from ayaka.configs.tokenizer import TokenizerConfig
from ayaka.tokenizers.contracts import LoadedTokenizer, TokenizerCapabilities
from ayaka.tokenizers.decoding.byte import verify_byte_table
from ayaka.tokenizers.service import TokenizerService

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ayaka.tokenizers.ports import DecodeStreamLike

BOS, EOS, IM_START, IM_END = 256, 257, 258, 259
SPECIALS = {"<s>": BOS, "</s>": EOS, "<|im_start|>": IM_START, "<|im_end|>": IM_END}
MODEL_VOCAB = 300
_TABLE: tuple[bytes, ...] = (
    *(bytes([i]) for i in range(256)),
    *(s.encode() for s in SPECIALS),
)


class _Stream:
    """Incremental UTF-8 stream: returns ``None`` until a character completes."""

    def __init__(self, skip_special_tokens: bool) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._skip = skip_special_tokens

    def step(self, token_id: int) -> str | None:
        if token_id >= 256:
            return None if self._skip else next(k for k, v in SPECIALS.items() if v == token_id)
        text = self._decoder.decode(bytes([token_id]))
        return text or None


def _calls(message: Mapping[str, Any]) -> str:
    """A message's tool calls (and a tool turn's id and name) as short text, for the template."""
    parts = [
        f"<tool_call>{call['function']['name']}"
        f"{json.dumps(call['function']['arguments'], separators=(',', ':'))}</tool_call>"
        for call in message.get("tool_calls", ())
    ]
    if message.get("role") == "tool":
        parts.append(f"[{message.get('tool_call_id')}:{message.get('name')}]")
    return "".join(parts)


class FakeTokenizer:
    """Minimal ``TokenizerLike`` plus the byte-table extension."""

    def __init__(self, *, stream: bool = True, chunked: bool = False) -> None:
        self._stream = stream
        self._chunked = chunked
        self.byte_path_verified = False
        self.encode_calls = 0

    # -- metadata --
    @property
    def name_or_path(self) -> str:
        return "fake"

    def fingerprint(self) -> str:
        return "fake-fp"

    @property
    def eos_token_id(self) -> int | None:
        return EOS

    @property
    def eos_token_ids(self) -> frozenset[int]:
        return frozenset({EOS, IM_END})

    @property
    def bos_token_id(self) -> int | None:
        return BOS

    @property
    def pad_token_id(self) -> int | None:
        return None

    @property
    def all_special_ids(self) -> frozenset[int]:
        return frozenset(SPECIALS.values())

    @property
    def all_special_tokens(self) -> tuple[str, ...]:
        return tuple(SPECIALS)

    @property
    def vocab_size(self) -> int:
        return 260

    @property
    def max_token_id(self) -> int:
        return 259

    @property
    def valid_token_ids(self) -> frozenset[int]:
        return frozenset(range(260))

    @property
    def max_chars_per_token(self) -> int:
        return 12

    @property
    def is_fast(self) -> bool:
        return True

    @property
    def supports_chunked_encode(self) -> bool:
        return self._chunked

    # -- encoding --
    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        self.encode_calls += 1
        ids = list(text.encode())
        return [BOS, *ids] if add_special_tokens else ids

    def encode_batch(
        self, texts: Sequence[str], *, add_special_tokens: bool = True
    ) -> list[list[int]]:
        return [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        add_generation_prompt: bool = True,
        chat_template: str | None = None,
        **kwargs: Any,
    ) -> str:
        del chat_template, kwargs
        # Tool names and calls are written briefly, so a test can see they reached the prompt.
        names = ",".join(str(tool.get("function", tool).get("name")) for tool in tools or ())
        head = f"<|im_start|>tools\n{names}<|im_end|>\n" if names else ""
        body = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}{_calls(m)}<|im_end|>\n" for m in messages
        )
        return head + body + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    # -- decoding --
    def decode(
        self,
        ids: Sequence[int] | int,
        *,
        skip_special_tokens: bool = False,
        spaces_between_special_tokens: bool = True,
    ) -> str:
        del spaces_between_special_tokens
        sequence = [ids] if isinstance(ids, int) else list(ids)
        return b"".join(
            _TABLE[i] for i in sequence if not (skip_special_tokens and i >= 256)
        ).decode("utf-8", errors="replace")

    def convert_ids_to_tokens(
        self, ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> list[str]:
        return [self.decode([i], skip_special_tokens=skip_special_tokens) for i in ids]

    def convert_tokens_to_string(self, tokens: Sequence[str]) -> str:
        return "".join(tokens)

    def new_decode_stream(self, *, skip_special_tokens: bool = False) -> DecodeStreamLike | None:
        return _Stream(skip_special_tokens) if self._stream else None

    # -- vocabulary --
    def get_vocab(self) -> dict[str, int]:
        return {str(i): i for i in range(260)}

    def get_added_vocab(self) -> dict[str, int]:
        return dict(SPECIALS)

    def get_decoded_vocab(self) -> tuple[bytes, ...]:
        return _TABLE

    def token_bytes_table(self) -> tuple[bytes, ...]:
        return _TABLE

    def verify_byte_path(self, samples: Sequence[str] = ()) -> bool:
        self.byte_path_verified = verify_byte_table(self, samples or ("héllo wörld", "日本語 🚀"))
        return self.byte_path_verified


class RenderedFake(FakeTokenizer):
    """Also turns control tokens in rendered prompts into special ids."""

    def encode_rendered_batch(self, texts: Sequence[str]) -> list[list[int]]:
        out: list[list[int]] = []
        for text in texts:
            ids: list[int] = []
            i = 0
            while i < len(text):
                for special, ident in SPECIALS.items():
                    if text.startswith(special, i):
                        ids.append(ident)
                        i += len(special)
                        break
                else:
                    ids.extend(text[i].encode())
                    i += 1
            out.append(ids)
        return out


def loaded_for(
    tokenizer: FakeTokenizer, *, stream: bool = True, byte: bool = True
) -> LoadedTokenizer:
    return LoadedTokenizer(
        tokenizer,
        TokenizerCapabilities(
            "hf",
            text_encode=True,
            text_decode=True,
            chat=True,
            decode_stream=stream,
            byte_decode=byte,
        ),
    )


def _service(tokenizer: FakeTokenizer) -> TokenizerService:
    return TokenizerService(
        TokenizerConfig("fake", mode="hf"),
        max_model_len=64,
        model_vocab_size=MODEL_VOCAB,
        loaded=loaded_for(tokenizer),
    )


# Zero-argument factories for ProcessTokenizerPool: a spawned child imports them by name,
# so they must stay importable module-level functions.


def make_service() -> TokenizerService:
    """A service over the plain fake (the stream decoding path)."""
    return _service(FakeTokenizer())


def make_rendered_service() -> TokenizerService:
    """A service whose tokenizer also turns control tokens in rendered prompts into ids."""
    return _service(RenderedFake())


OPEN_CALL, CLOSE_CALL = 260, 261
_CALL_MARKERS = {"<tool_call>": OPEN_CALL, "</tool_call>": CLOSE_CALL}
_TOOL_TABLE: tuple[bytes, ...] = (*_TABLE, b"<tool_call>", b"</tool_call>")


class ToolTokenizer(FakeTokenizer):
    """The byte tokenizer plus the Hermes tool-call markers as ordinary (not special) tokens.

    ``<tool_call>`` is id 260 and ``</tool_call>`` id 261, so a parser and a grammar can match
    them by id, as with Qwen and Hermes vocabularies.
    """

    @property
    def vocab_size(self) -> int:
        return 262

    @property
    def max_token_id(self) -> int:
        return 261

    @property
    def valid_token_ids(self) -> frozenset[int]:
        return frozenset(range(262))

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        self.encode_calls += 1
        ids: list[int] = []
        i = 0
        while i < len(text):
            for marker, ident in _CALL_MARKERS.items():
                if text.startswith(marker, i):
                    ids.append(ident)
                    i += len(marker)
                    break
            else:
                ids.extend(text[i].encode())
                i += 1
        return [BOS, *ids] if add_special_tokens else ids

    def decode(
        self,
        ids: Sequence[int] | int,
        *,
        skip_special_tokens: bool = False,
        spaces_between_special_tokens: bool = True,
    ) -> str:
        del spaces_between_special_tokens
        sequence = [ids] if isinstance(ids, int) else list(ids)
        special = set(SPECIALS.values())
        return b"".join(
            _TOOL_TABLE[i] for i in sequence if not (skip_special_tokens and i in special)
        ).decode("utf-8", errors="replace")

    def new_decode_stream(self, *, skip_special_tokens: bool = False) -> DecodeStreamLike | None:
        return None

    def get_vocab(self) -> dict[str, int]:
        return {**{str(i): i for i in range(260)}, **_CALL_MARKERS}

    def get_decoded_vocab(self) -> tuple[bytes, ...]:
        return _TOOL_TABLE

    def token_bytes_table(self) -> tuple[bytes, ...]:
        return _TOOL_TABLE
