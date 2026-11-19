"""Prefix reuse in the cache manager: match, attach, register, share, free and evict."""

from __future__ import annotations

import random

import pytest

from ayaka.core.errors import EngineError
from ayaka.core.types import KVLayoutKind, RequestId
from ayaka.kvcache.manager import KVCacheManager
from ayaka.kvcache.prefix.hash_cache import HashPrefixCache
from ayaka.kvcache.prefix.hashing import ChainRoot
from ayaka.kvcache.storage.spec import KVGroup, KVSpec
from ayaka.utils.dtypes import DType

pytestmark = pytest.mark.unit

BLOCK = 4


def _rid(name: str | int) -> RequestId:
    return RequestId(str(name))


def _group(layers: tuple[int, ...]) -> KVGroup:
    spec = KVSpec(
        num_layers=len(layers),
        num_kv_heads=1,
        head_dim=4,
        block_size=BLOCK,
        dtype=DType.FP16,
        layout=KVLayoutKind.NHD,
    )
    return KVGroup(layer_ids=layers, spec=spec)


def _cache(model: str = "m") -> HashPrefixCache:
    return HashPrefixCache(ChainRoot(model_id=model, dtype="fp16", codec="raw", block_size=BLOCK))


def _manager(
    blocks: tuple[int, ...] = (17,),
    *,
    max_requests: int = 4,
    max_blocks: int = 8,
    cache: HashPrefixCache | None = None,
) -> KVCacheManager:
    groups = [_group((i,)) for i in range(len(blocks))]
    return KVCacheManager(
        groups,
        blocks,
        max_requests=max_requests,
        max_blocks_per_request=max_blocks,
        prefix=cache or _cache(),
    )


def _run_prompt(
    manager: KVCacheManager, name: str, prompt: list[int], salt: str | None = None
) -> int:
    """Admit, prefill and register a prompt the way the scheduler does; return reused tokens."""
    request_id = _rid(name)
    match = manager.match_prefix(prompt, salt)
    assert match is not None
    assert manager.allocate(request_id, len(prompt) - match.num_tokens, match)
    manager.commit(request_id, len(prompt) - match.num_tokens)
    registered, full = manager.cacheable_blocks(request_id)
    manager.cache_blocks(request_id, prompt[registered * BLOCK : full * BLOCK])
    return match.num_tokens


_PROMPT = list(range(10, 23))  # 13 tokens: three full blocks and one partial


def test_a_second_request_with_the_same_prompt_reuses_its_full_blocks() -> None:
    manager = _manager()

    assert _run_prompt(manager, "a", _PROMPT) == 0
    assert _run_prompt(manager, "b", _PROMPT) == 12

    a, b = manager.blocks(_rid("a"))[0], manager.blocks(_rid("b"))[0]
    assert a[:3] == b[:3]
    assert a[3] != b[3]  # the partial last block is private
    assert manager.num_committed(_rid("b")) == 13
    assert all(manager._pools[0].ref_count(block) == 2 for block in a[:3])
    manager.check_integrity()


def test_without_a_prefix_cache_there_is_no_match() -> None:
    manager = KVCacheManager(
        [_group((0,))], [9], max_requests=2, max_blocks_per_request=4, prefix=None
    )

    assert manager.match_prefix(_PROMPT, None) is None
    assert not manager.prefix_caching


def test_a_prompt_of_whole_blocks_leaves_the_last_block_to_compute() -> None:
    manager = _manager()
    prompt = list(range(8))  # exactly two blocks

    _run_prompt(manager, "a", prompt)
    match = manager.match_prefix(prompt, None)

    assert match is not None
    assert match.num_blocks == 1
    assert match.num_tokens == 4
    assert len(match.digests) == 2


def test_a_partial_last_block_is_never_registered() -> None:
    manager = _manager()
    _run_prompt(manager, "a", _PROMPT)
    last = manager.blocks(_rid("a"))[0][3]

    assert manager._pools[0].digest_of(last) is None
    assert manager.cacheable_blocks(_rid("a")) == (3, 3)


@pytest.mark.parametrize(
    ("salt", "model"), [("other", "m"), (None, "other-model")], ids=["salt", "model"]
)
def test_a_different_salt_or_model_matches_nothing(salt: str | None, model: str) -> None:
    cache = _cache()
    manager = _manager(cache=cache)
    _run_prompt(manager, "a", _PROMPT, salt="tenant")

    if model == "m":
        match = manager.match_prefix(_PROMPT, salt)
    else:
        match = _cache(model).match(manager._pools, _PROMPT, "tenant")
    assert match is not None
    assert match.num_blocks == 0
    same = manager.match_prefix(_PROMPT, "tenant")
    assert same is not None
    assert same.num_blocks == 3


def test_freeing_one_of_two_sharing_requests_keeps_the_other_valid() -> None:
    manager = _manager()
    _run_prompt(manager, "a", _PROMPT)
    _run_prompt(manager, "b", _PROMPT)
    shared = manager.blocks(_rid("b"))[0][:3]

    manager.free(_rid("a"))

    manager.check_integrity()
    assert all(manager._pools[0].ref_count(block) == 1 for block in shared)
    assert manager.blocks(_rid("b"))[0][:3] == shared
    manager.free(_rid("b"))
    manager.check_integrity()
    stats = manager.stats()
    assert stats.cached_blocks == (3,)
    assert stats.free_blocks == stats.usable_blocks


def test_a_freed_prefix_is_still_matched_until_its_blocks_are_reused() -> None:
    manager = _manager()
    _run_prompt(manager, "a", _PROMPT)
    manager.free(_rid("a"))

    assert _run_prompt(manager, "b", _PROMPT) == 12
    manager.check_integrity()


def test_eviction_takes_unregistered_blocks_first_then_cached_leaves_before_roots() -> None:
    manager = _manager((9,), max_blocks=8)  # 8 usable blocks
    _run_prompt(manager, "a", _PROMPT)  # blocks 1..4; 1..3 registered
    root, middle, leaf, partial = manager.blocks(_rid("a"))[0]
    manager.free(_rid("a"))
    pool = manager._pools[0]

    queue = pool.free_queue()
    assert queue[0] == partial  # unregistered: reused first
    assert queue[-3:] == [leaf, middle, root]  # a cached chain is eaten from the leaf

    assert manager.allocate(_rid("filler"), 6 * BLOCK)  # the 4 never-used blocks, partial, leaf
    assert pool.digest_of(root) is not None
    assert pool.digest_of(leaf) is None
    assert manager.stats().evictions == 1
    match = manager.match_prefix(_PROMPT, None)
    assert match is not None
    assert match.num_blocks == 2  # root and middle survive


def test_a_second_turn_extends_the_match_over_generated_blocks() -> None:
    manager = _manager()
    request = _rid("a")
    prompt = list(range(6))
    match = manager.match_prefix(prompt, None)
    assert manager.allocate(request, len(prompt), match)
    manager.commit(request, len(prompt))
    generated = [50, 51, 52, 53, 54]
    sequence = prompt + generated
    for position in range(len(prompt), len(sequence)):
        assert manager.allocate(request, 1)
        manager.commit(request, 1)
        registered, full = manager.cacheable_blocks(request)
        if full > registered:
            manager.cache_blocks(request, sequence[registered * BLOCK : full * BLOCK])
        assert full == (position + 1) // BLOCK
    manager.free(request)

    second = manager.match_prefix([*sequence, 99, 98], None)

    assert second is not None
    assert second.num_blocks == 2
    manager.check_integrity()


def test_the_same_digest_registered_twice_stays_private_to_the_second_request() -> None:
    manager = _manager()
    for name in ("a", "b"):
        match = manager.match_prefix(_PROMPT, None)
        assert match is not None
        assert manager.allocate(_rid(name), len(_PROMPT), match)
    for name in ("a", "b"):
        manager.commit(_rid(name), len(_PROMPT))
    assert manager.cache_blocks(_rid("a"), _PROMPT[:12]) == 3
    assert manager.cache_blocks(_rid("b"), _PROMPT[:12]) == 0

    pool = manager._pools[0]
    assert all(pool.digest_of(block) is None for block in manager.blocks(_rid("b"))[0])
    manager.check_integrity()


def test_caching_a_block_that_is_not_full_is_an_error() -> None:
    manager = _manager()
    match = manager.match_prefix(_PROMPT, None)
    assert manager.allocate(_rid("a"), 13, match)
    manager.commit(_rid("a"), 7)

    with pytest.raises(EngineError, match="only 1 blocks are full"):
        manager.cache_blocks(_rid("a"), _PROMPT[:8])


def test_a_request_admitted_without_a_match_never_registers() -> None:
    manager = _manager()
    assert manager.allocate(_rid("a"), 13)
    manager.commit(_rid("a"), 13)

    assert manager.cacheable_blocks(_rid("a")) == (0, 0)
    assert manager.cache_blocks(_rid("a"), _PROMPT[:12]) == 0
    assert manager.stats().cached_blocks == (0,)


def test_attaching_a_prefix_without_room_changes_nothing() -> None:
    manager = _manager((7,), max_blocks=6)  # 6 usable blocks
    _run_prompt(manager, "a", _PROMPT)
    manager.free(_rid("a"))
    assert manager.allocate(_rid("filler"), 2 * BLOCK)  # takes the 2 never-used blocks
    before = (manager._pools[0].free_queue(), manager.table.rows.tolist(), manager.stats())

    long_prompt = [*_PROMPT[:12], *range(200, 220)]  # 3 cached blocks, then 5 new ones
    match = manager.match_prefix(long_prompt, None)
    assert match is not None
    assert match.num_blocks == 3
    # Attaching revives 3 cached blocks; 1 free block is left for 5 new ones.
    assert not manager.allocate(_rid("b"), len(long_prompt) - match.num_tokens, match)

    assert (manager._pools[0].free_queue(), manager.table.rows.tolist(), manager.stats()) == before


def test_a_prefix_cannot_be_attached_to_a_request_that_holds_blocks() -> None:
    manager = _manager()
    assert manager.allocate(_rid("a"), 4)
    match = manager.match_prefix(_PROMPT, None)

    with pytest.raises(EngineError, match="already holds blocks"):
        manager.allocate(_rid("a"), 4, match)


def test_stats_count_admitted_lookups_and_reused_tokens() -> None:
    manager = _manager()
    _run_prompt(manager, "a", _PROMPT)
    _run_prompt(manager, "b", _PROMPT)

    stats = manager.stats()
    assert stats.lookups == 2
    assert stats.hit_tokens == 12


def test_groups_share_matches_and_a_one_sided_digest_is_an_integrity_error() -> None:
    manager = _manager((17, 17))
    _run_prompt(manager, "a", _PROMPT)
    _run_prompt(manager, "b", _PROMPT)
    assert manager.blocks(_rid("b"))[1][:3] == manager.blocks(_rid("a"))[1][:3]
    manager.check_integrity()
    manager.free(_rid("a"))
    manager.free(_rid("b"))

    # Register a stray digest in group 1 only.
    root = manager._prefix.root(None)  # type: ignore[union-attr]  # set by _manager
    digest = manager._prefix.digests(root, _PROMPT)[0]  # type: ignore[union-attr]
    pool0 = manager._pools[0]
    holder = pool0.lookup(digest)
    assert holder is not None
    pool0._by_digest.pop(digest)
    pool0._digest[holder] = None

    with pytest.raises(EngineError, match="some KV groups only"):
        manager.match_prefix(_PROMPT, None)


def test_random_alloc_match_free_sequences_keep_the_books_clean() -> None:
    rng = random.Random(7)  # noqa: S311  # a seeded simulation, not security
    manager = _manager((40, 40), max_requests=6, max_blocks=10)
    prompts = [[rng.randrange(5) for _ in range(rng.randrange(1, 30))] for _ in range(6)]
    live: dict[RequestId, list[int]] = {}
    for step in range(400):
        action = rng.random()
        if action < 0.45 and len(live) < 6:
            request = _rid(step)
            base = rng.choice(prompts)
            prompt = base[: rng.randrange(1, len(base) + 1)]
            match = manager.match_prefix(prompt, rng.choice([None, "s"]))
            assert match is not None
            if manager.allocate(request, len(prompt) - match.num_tokens, match):
                manager.commit(request, len(prompt) - match.num_tokens)
                live[request] = list(prompt)
        elif action < 0.75 and live:
            request = rng.choice(list(live))
            if manager.allocate(request, 1):
                manager.commit(request, 1)
                live[request].append(rng.randrange(5))
        elif live:
            request = rng.choice(list(live))
            manager.free(request)
            del live[request]
        for request, tokens in live.items():
            registered, full = manager.cacheable_blocks(request)
            if full > registered:
                manager.cache_blocks(request, tokens[registered * BLOCK : full * BLOCK])
        manager.check_integrity()
    for request in list(live):
        manager.free(request)
    manager.check_integrity()
    stats = manager.stats()
    assert stats.free_blocks == stats.usable_blocks
