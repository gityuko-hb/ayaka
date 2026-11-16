"""The block pool: free-queue order, eviction, references, and a model-based check."""

from __future__ import annotations

import random
import subprocess
import sys

import pytest

from ayaka.core.errors import CapacityError, EngineError
from ayaka.kvcache.blocks import Block, BlockPool
from ayaka.kvcache.storage.spec import PADDING_BLOCK_ID

pytestmark = pytest.mark.unit


def _digest(n: int) -> bytes:
    return n.to_bytes(16, "little")


# -- the basics -----


def test_a_new_pool_queues_every_block_but_the_padding_block_in_order() -> None:
    pool = BlockPool(6)

    assert pool.free_queue() == [1, 2, 3, 4, 5]
    assert pool.free_count == 5
    assert pool.num_blocks == 6
    pool.check_integrity()


def test_block_zero_is_never_allocated_and_has_no_bookkeeping() -> None:
    pool = BlockPool(3)

    assert [pool.allocate(), pool.allocate()] == [1, 2]
    with pytest.raises(CapacityError):
        pool.allocate()
    for call in (pool.ref_count, pool.digest_of, pool.is_free, pool.touch, pool.free, pool.block):
        with pytest.raises(ValueError, match="padding block"):
            call(PADDING_BLOCK_ID)


@pytest.mark.parametrize("block_id", [-1, 6, 100])
def test_an_id_outside_the_pool_is_refused(block_id: int) -> None:
    with pytest.raises(ValueError, match="outside"):
        BlockPool(6).ref_count(block_id)


def test_a_pool_needs_the_padding_block_and_one_more() -> None:
    with pytest.raises(ValueError, match="num_blocks"):
        BlockPool(1)
    assert BlockPool(2).free_queue() == [1]


def test_allocate_takes_the_head_with_one_reference() -> None:
    pool = BlockPool(5)

    block = pool.allocate()

    assert block == 1
    assert pool.ref_count(1) == 1
    assert not pool.is_free(1)
    assert pool.free_queue() == [2, 3, 4]
    assert pool.free_count == 3


def test_the_block_view_reads_through_to_the_pool() -> None:
    pool = BlockPool(4)
    block_id = pool.allocate()
    pool.register(block_id, _digest(9))

    view = pool.block(block_id)

    assert isinstance(view, Block)
    assert (view.id, view.ref_count, view.digest, view.is_free) == (block_id, 1, _digest(9), False)


# -- returning blocks -----


def test_an_unregistered_block_returns_to_the_front() -> None:
    pool = BlockPool(6)
    first, second = pool.allocate(), pool.allocate()

    pool.free(first)
    pool.free(second)

    assert pool.free_queue() == [second, first, 3, 4, 5]


def test_a_registered_block_returns_to_the_back() -> None:
    pool = BlockPool(6)
    block = pool.allocate()
    pool.register(block, _digest(1))

    pool.free(block)

    assert pool.free_queue() == [2, 3, 4, 5, block]
    assert pool.cached_count == 1


def test_an_unregistered_block_returns_in_front_of_cached_ones() -> None:
    pool = BlockPool(6)
    cached, plain = pool.allocate(), pool.allocate()
    pool.register(cached, _digest(1))
    pool.free(cached)

    pool.free(plain)

    assert pool.free_queue()[0] == plain
    assert pool.free_queue()[-1] == cached


def test_a_chain_is_returned_last_block_first() -> None:
    pool = BlockPool(8)
    chain = [pool.allocate() for _ in range(3)]

    pool.free_chain(chain)

    # Pushed to the front in reverse, so the chain's first block is reused first.
    assert pool.free_queue() == [*chain, 4, 5, 6, 7]


def test_a_cached_chain_loses_its_leaf_before_its_root() -> None:
    pool = BlockPool(8)
    chain = [pool.allocate() for _ in range(3)]
    for position, block in enumerate(chain):
        pool.register(block, _digest(position))

    pool.free_chain(chain)

    root, middle, leaf = chain
    assert pool.free_queue()[-3:] == [leaf, middle, root]
    # Eat the queue from the head: unregistered blocks first, then the leaf, never the root first.
    reused = [pool.allocate() for _ in range(4 + 1)]
    assert reused[-1] == leaf
    assert pool.lookup(_digest(0)) == root


def test_a_shared_block_rejoins_only_when_its_last_reference_goes() -> None:
    pool = BlockPool(4)
    block = pool.allocate()
    pool.touch(block)

    pool.free(block)

    assert pool.ref_count(block) == 1
    assert not pool.is_free(block)
    pool.free(block)
    assert pool.is_free(block)
    assert pool.free_queue()[0] == block


def test_freeing_an_unreferenced_block_is_an_error() -> None:
    pool = BlockPool(4)

    with pytest.raises(EngineError, match="already free"):
        pool.free(1)


# -- registering and evicting -----


def test_a_cached_block_can_be_matched_and_touched_back_out_of_the_queue() -> None:
    pool = BlockPool(5)
    block = pool.allocate()
    pool.register(block, _digest(7))
    pool.free(block)

    assert pool.lookup(_digest(7)) == block
    pool.touch(block)

    assert pool.ref_count(block) == 1
    assert not pool.is_free(block)
    assert block not in pool.free_queue()
    assert pool.lookup(_digest(7)) == block
    pool.check_integrity()


def test_allocating_a_cached_block_evicts_it() -> None:
    pool = BlockPool(3)
    first = pool.allocate()
    second = pool.allocate()
    pool.register(first, _digest(1))
    pool.register(second, _digest(2))
    pool.free_chain([first, second])

    reused = pool.allocate()

    assert reused == second
    assert pool.lookup(_digest(2)) is None
    assert pool.digest_of(reused) is None
    assert pool.lookup(_digest(1)) == first
    assert pool.evictions == 1
    pool.check_integrity()


def test_registering_a_digest_that_exists_leaves_the_block_private() -> None:
    pool = BlockPool(4)
    owner, other = pool.allocate(), pool.allocate()

    assert pool.register(owner, _digest(1))
    assert not pool.register(other, _digest(1))

    assert pool.lookup(_digest(1)) == owner
    assert pool.digest_of(other) is None


def test_registering_needs_a_reference_and_happens_once() -> None:
    pool = BlockPool(4)
    block = pool.allocate()

    with pytest.raises(EngineError, match="not referenced"):
        pool.register(2, _digest(1))
    pool.register(block, _digest(1))
    with pytest.raises(EngineError, match="already registered"):
        pool.register(block, _digest(2))


def test_touching_a_block_nobody_holds_and_nobody_queued_is_an_error() -> None:
    pool = BlockPool(4)
    block = pool.allocate()
    pool._ref[block] = 0  # corrupt: unreferenced and not queued

    with pytest.raises(EngineError, match="not free"):
        pool.touch(block)


# -- check_integrity -----


def _corrupt_ref(pool: BlockPool) -> None:
    pool._ref[2] = -1


def _corrupt_count(pool: BlockPool) -> None:
    pool._free_count += 1


def _corrupt_leak(pool: BlockPool) -> None:
    pool._unlink(2)


def _corrupt_free_with_refs(pool: BlockPool) -> None:
    pool._ref[2] = 1


def _corrupt_digest_table(pool: BlockPool) -> None:
    pool._by_digest[_digest(1)] = 3


def _corrupt_orphan_digest(pool: BlockPool) -> None:
    pool._digest[3] = _digest(5)


def _corrupt_link(pool: BlockPool) -> None:
    pool._next[1] = 4


def _corrupt_cycle(pool: BlockPool) -> None:
    pool._next[4] = 1


def _corrupt_padding(pool: BlockPool) -> None:
    pool._ref[PADDING_BLOCK_ID] = 1


@pytest.mark.parametrize(
    "corrupt",
    [
        _corrupt_ref,
        _corrupt_count,
        _corrupt_leak,
        _corrupt_free_with_refs,
        _corrupt_digest_table,
        _corrupt_orphan_digest,
        _corrupt_link,
        _corrupt_cycle,
        _corrupt_padding,
    ],
)
def test_check_integrity_names_a_corrupted_pool(corrupt: object) -> None:
    pool = BlockPool(5)
    pool.check_integrity()

    corrupt(pool)  # type: ignore[operator]  # each parametrised value is a function of the pool

    with pytest.raises(EngineError, match="inconsistent"):
        pool.check_integrity()


# -- the model: what a naive dict-and-list implementation does -----


class ReferencePool:
    """The naive version: a Python list for the queue, dicts for everything else."""

    def __init__(self, num_blocks: int) -> None:
        self.queue = list(range(1, num_blocks))
        self.ref = dict.fromkeys(range(1, num_blocks), 0)
        self.digest: dict[int, bytes] = {}
        self.table: dict[bytes, int] = {}
        self.evictions = 0

    def allocate(self) -> int:
        block = self.queue.pop(0)
        old = self.digest.pop(block, None)
        if old is not None:
            del self.table[old]
            self.evictions += 1
        self.ref[block] = 1
        return block

    def touch(self, block: int) -> None:
        if self.ref[block] == 0:
            self.queue.remove(block)
        self.ref[block] += 1

    def free_chain(self, chain: list[int]) -> None:
        for block in reversed(chain):
            self.ref[block] -= 1
            if self.ref[block] == 0:
                if block in self.digest:
                    self.queue.append(block)
                else:
                    self.queue.insert(0, block)

    def register(self, block: int, digest: bytes) -> bool:
        if digest in self.table:
            return False
        self.table[digest] = block
        self.digest[block] = digest
        return True


def _agree(pool: BlockPool, model: ReferencePool, num_blocks: int) -> None:
    assert pool.free_queue() == model.queue
    assert pool.free_count == len(model.queue)
    assert pool.evictions == model.evictions
    for block in range(1, num_blocks):
        assert pool.ref_count(block) == model.ref[block]
        assert pool.digest_of(block) == model.digest.get(block)
    for digest, block in model.table.items():
        assert pool.lookup(digest) == block
    pool.check_integrity()


@pytest.mark.parametrize("seed", range(8))
def test_the_pool_agrees_with_the_reference_model_on_random_operations(seed: int) -> None:
    rng = random.Random(seed)  # noqa: S311  # a seeded simulation, not security
    num_blocks = rng.choice([4, 9, 33])
    pool, model = BlockPool(num_blocks), ReferencePool(num_blocks)
    chains: list[list[int]] = []
    next_digest = 0

    for _ in range(1500):
        action = rng.choices(
            ["alloc", "free", "register", "share", "dup_register"], weights=[30, 25, 20, 15, 10]
        )[0]
        if action == "alloc":
            size = rng.randint(1, 4)
            if pool.free_count >= size:
                chain = [pool.allocate() for _ in range(size)]
                assert chain == [model.allocate() for _ in range(size)]
                chains.append(chain)
        elif action == "free" and chains:
            chain = chains.pop(rng.randrange(len(chains)))
            pool.free_chain(chain)
            model.free_chain(chain)
        elif action in ("register", "dup_register") and chains:
            block = rng.choice(rng.choice(chains))
            if model.digest.get(block) is None:
                if action == "dup_register" and model.table:
                    digest = rng.choice(list(model.table))
                else:
                    next_digest += 1
                    digest = _digest(next_digest)
                assert pool.register(block, digest) == model.register(block, digest)
        elif action == "share" and model.table:
            digest = rng.choice(list(model.table))
            block = pool.lookup(digest)
            assert block == model.table[digest]
            assert block is not None
            pool.touch(block)
            model.touch(block)
            chains.append([block])
        _agree(pool, model, num_blocks)

    for chain in chains:
        pool.free_chain(chain)
        model.free_chain(chain)
    _agree(pool, model, num_blocks)
    assert pool.free_count == num_blocks - 1


def test_a_thousand_requests_leave_the_pool_whole() -> None:
    rng = random.Random(0)  # noqa: S311  # a seeded simulation, not security
    pool = BlockPool(4000)
    held: list[list[int]] = []

    for _ in range(1000):
        held.append([pool.allocate() for _ in range(rng.randint(1, 3))])
        if rng.random() < 0.5:
            pool.free_chain(held.pop(rng.randrange(len(held))))
    for chain in held:
        pool.free_chain(chain)

    assert pool.free_count == 3999
    pool.check_integrity()


def test_importing_the_block_pool_does_not_import_torch() -> None:
    program = (
        "import sys\n"
        "import ayaka.kvcache.blocks\n"
        "import ayaka.kvcache.table\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )

    result = subprocess.run(  # noqa: S603  # fixed argv, no input from outside
        [sys.executable, "-I", "-c", program], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
