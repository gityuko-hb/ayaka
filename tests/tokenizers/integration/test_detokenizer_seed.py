"""A detokenizer seeded with the end of the prompt says what one seeded with all of it says.

The suffix seed is compared with the full seed of the same classes (the seed sizes patched to
"the whole prompt"), on real tokenizers of the three layouts whose decoders keep state across
tokens: ByteLevel (GPT-2, Llama 3; the byte path), Metaspace (SentencePiece; the ``DecodeStream``
path) and byte fallback (Llama 2, where a character the vocabulary lacks is spelled byte by byte,
so a prompt can end inside a character). Prompts are longer than the largest seed window.
"""

from __future__ import annotations

import random
import threading
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.configs.tokenizer import DetokenizeParams
from ayaka.tokenizers.contracts import EncodeOptions, TextInput
from ayaka.tokenizers.decoding import byte, incremental
from ayaka.tokenizers.decoding.incremental import MAX_SEED_WINDOW, IncrementalDetokenizer
from ayaka.tokenizers.factory import DefaultTokenizerFactory
from ayaka.tokenizers.service import TokenizerService
from real_tokenizers import KINDS, build_hf_dir_of, hf_config

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.tokenizers.contracts import LoadedTokenizer

pytestmark = pytest.mark.integration

_WORDS = ("hello", "world", "héllo", "wörld", "ça", "va?", "中文", "текст", "🙂", "ok,", "tiny")


@pytest.fixture(scope="module", params=KINDS)
def loaded(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Any:
    kind = str(request.param)
    directory = build_hf_dir_of(tmp_path_factory.mktemp(kind) / "tokenizer", kind)
    return DefaultTokenizerFactory.from_config(hf_config(directory))


def _text(rng: random.Random, words: int) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(words))


def _run(
    loaded: LoadedTokenizer,
    backend: str,
    prompt: Sequence[int],
    output: Sequence[int],
    skip: bool,
) -> tuple[str, ...] | type[BaseException]:
    """Every delta and the finish of one request, or the type of error it ended with."""
    params = DetokenizeParams(skip_special_tokens=skip)
    tokenizer = loaded.tokenizer
    assert tokenizer is not None
    try:
        detok = IncrementalDetokenizer.create(
            tokenizer,
            params,
            prompt_token_ids=prompt,
            backend=backend,  # type: ignore[arg-type]  # one of the backends, by name
        )
        pieces = [detok.update([token]).delta for token in output]
        return (*pieces, detok.finish().delta)
    except Exception as exc:  # the native DecodeStream raises a bare Exception
        return type(exc)


def _full_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every path replay the whole prompt, as it did before the seeds were bounded."""
    monkeypatch.setattr(byte, "SEED_TOKENS", 10**9)
    monkeypatch.setattr(incremental, "SEED_WINDOW", 10**9)
    monkeypatch.setattr(incremental, "MAX_SEED_WINDOW", 10**9)


@pytest.mark.parametrize("backend", ["auto", "stream", "window"])
def test_a_suffix_seed_decodes_like_the_whole_prompt(
    loaded: LoadedTokenizer, backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = loaded.tokenizer
    assert tokenizer is not None
    rng = random.Random(29)  # noqa: S311  # reproducible cases, not security
    valid = sorted(tokenizer.valid_token_ids)
    cases: list[tuple[list[int], list[int], bool]] = []
    for _ in range(12):
        prompt = tokenizer.encode(_text(rng, 400), add_special_tokens=False)
        assert len(prompt) > MAX_SEED_WINDOW
        cut = rng.randint(1, 4)  # the prompt may end inside a character
        spelled = tokenizer.encode(_text(rng, 30), add_special_tokens=False)
        cases.append((prompt + spelled[:cut], spelled[cut:], True))
        random_ids = [rng.choice(valid) for _ in range(40)]
        cases.append((prompt, random_ids, rng.random() < 0.5))
    if backend == "auto":
        cases = [(p, o, True) for p, o, _ in cases]  # the byte path needs skipped specials

    seeded = [_run(loaded, backend, *case) for case in cases]
    _full_seed(monkeypatch)
    whole = [_run(loaded, backend, *case) for case in cases]

    assert seeded == whole
    assert any(isinstance(result, tuple) for result in seeded)


def test_encoding_a_long_prompt_and_decoding_run_at_the_same_time(
    loaded: LoadedTokenizer,
) -> None:
    tokenizer = loaded.tokenizer
    assert tokenizer is not None
    service = TokenizerService(
        hf_config("unused"), max_model_len=200_000, model_vocab_size=1000, loaded=loaded
    )
    rng = random.Random(7)  # noqa: S311  # reproducible text, not security
    long_text = TextInput(_text(rng, 20_000), add_special_tokens=False)
    expected = service.encode(long_text, EncodeOptions(max_output_tokens=1)).token_ids
    prompt = tokenizer.encode(_text(rng, 50), add_special_tokens=False)
    output = tokenizer.encode(_text(rng, 50), add_special_tokens=False)
    params = DetokenizeParams(skip_special_tokens=True)

    def decode_once() -> str:
        detok = service.new_detokenizer(params, prompt_token_ids=tuple(prompt))
        return "".join(detok.update([token]).delta for token in output) + detok.finish().delta

    reference = decode_once()
    failures: list[BaseException] = []
    decoded: list[str] = []
    stop = threading.Event()

    def encoder() -> None:
        try:
            while not stop.is_set():
                ids = service.encode(long_text, EncodeOptions(max_output_tokens=1)).token_ids
                assert ids == expected
        except BaseException as exc:  # reported below, in the test thread
            failures.append(exc)

    thread = threading.Thread(target=encoder)
    thread.start()
    try:
        decoded.extend(decode_once() for _ in range(20))
    finally:
        stop.set()
        thread.join()
        service.close()

    assert failures == []
    assert decoded == [reference] * 20
