"""Gate 2 tests for NgramIndex, LookupProposer, and TokenRecyclingProposer."""

from __future__ import annotations

import random

import pytest

from ayaka.configs.model import ModelConfig
from ayaka.core.types import RequestId
from ayaka.speculative.ngram import LookupProposer, NgramIndex, naive_ngram_match
from ayaka.speculative.token_recycling import TokenRecyclingProposer

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ngram_index_matches_naive_scan_on_random_and_repetitive_sequences(seed: int) -> None:
    rng = random.Random(seed)  # noqa: S311
    # Mix random tokens with repeated blocks so matches of lengths 3..32 occur.
    block = [rng.randrange(500) for _ in range(40)]
    seq: list[int] = []
    for i in range(40):
        if i % 4 == 0:
            seq.extend(block[: rng.randint(5, 40)])
        else:
            seq.extend(rng.randrange(30) for _ in range(25))

    index = NgramIndex(seq[:100])
    for step in range(100, len(seq)):
        suffix = seq[max(0, step - 40) : step]
        got = index.match(suffix, max_k=8, min_match=3, max_match=32, max_candidates=4)
        exp = naive_ngram_match(
            seq[:step], suffix, max_k=8, min_match=3, max_match=32, max_candidates=4
        )
        assert got == exp
        index.extend([seq[step]])


def test_ngram_index_memory_at_100k_tokens_is_under_128_bytes_per_token() -> None:
    rng = random.Random(7)  # noqa: S311
    tokens = [rng.randrange(32000) for _ in range(100_000)]
    index = NgramIndex(tokens)
    bytes_per_token = index.memory_bytes() / len(tokens)
    assert bytes_per_token <= 128.0, f"got {bytes_per_token:.1f} B/token"


def test_lookup_proposer_prediction_and_ban_list_filtering() -> None:
    proposer = LookupProposer(min_match=3, max_match=32, candidates=4, accept_prediction=True)
    cfg = ModelConfig(
        architecture="LlamaForCausalLM",
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=4,
        head_dim=16,
        intermediate_size=128,
        max_position_embeddings=512,
        hidden_act="silu",
        norm_eps=1e-5,
        tie_word_embeddings=True,
        rope_theta=10000.0,
    )
    assert proposer.supports(cfg) is None

    rid = RequestId("req-1")
    prompt = [10, 11, 12, 13, 14, 15, 16]
    prediction = [1, 2, 3, 99, 4, 5, 6, 7]
    proposer.on_admit(rid, prompt, prediction=prediction, banned_tokens=(99,))

    # Before any output, prediction seeds from index 0 and truncates before banned token 99.
    p0 = proposer.propose(rid, max_k=8)
    assert p0.tokens == (1, 2, 3)
    assert p0.source == "prediction"

    # Output drifts then re-aligns with prediction [4, 5, 6] -> proposes 7.
    proposer.on_commit(rid, [1, 2, 3, 50, 4, 5, 6])
    p1 = proposer.propose(rid, max_k=4)
    assert p1.tokens == (7,)
    assert p1.source == "prediction"

    # Tail matches prompt [10, 11, 12, 13] -> proposes [14, 15, 16].
    proposer.on_commit(rid, [10, 11, 12, 13])
    p2 = proposer.propose(rid, max_k=3)
    assert p2.tokens == (14, 15, 16)
    assert p2.source == "lookup"
    assert p2.confidence == 4

    proposer.release(rid)
    assert proposer.propose(rid, max_k=4).tokens == ()


def test_token_recycling_proposer_tracks_transitions() -> None:
    proposer = TokenRecyclingProposer(width=4)
    rid = RequestId("tr-1")
    proposer.on_admit(rid, [1, 2, 3, 4, 5], banned_tokens=(5,))
    proposer.on_commit(rid, [2])
    prop = proposer.propose(rid, max_k=4)
    # 2 -> 3 -> 4, stops before banned token 5
    assert prop.tokens == (3, 4)
    proposer.release(rid)
