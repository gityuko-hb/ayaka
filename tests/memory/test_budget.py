"""The budget plan on a table of machines and models, its monotonicity, and the ledger."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from typing import cast

import pytest

from ayaka.core.errors import CapacityError
from ayaka.memory.budget import (
    BUDGET_OWNERS,
    BudgetInputs,
    DraftCost,
    ExpertCost,
    KVCost,
    MemoryBudget,
    plan_budget,
)

pytestmark = pytest.mark.unit

MIB = 2**20

# The reference model: 28 layers, 2 KV heads, head_dim 128, fp16, block of 16 tokens.
TOKEN_BYTES = 2 * 28 * 2 * 128 * 2
BLOCK_BYTES = TOKEN_BYTES * 16
DENSE_KV = KVCost(bytes_per_block=BLOCK_BYTES, block_size=16)
WEIGHTS_FP16 = 2861 * MIB


def _inputs(**overrides: object) -> BudgetInputs:
    fields: dict[str, object] = {
        "baseline_bytes": 3700 * MIB,
        "memory_ratio": 0.9,
        "weights_bytes": WEIGHTS_FP16,
        "kv": (DENSE_KV,),
        "kv_reserve_tokens": 1024,
    }
    fields.update(overrides)
    return BudgetInputs(**fields)  # type: ignore[arg-type]  # the table passes valid field types


def _expert_inputs(**overrides: object) -> BudgetInputs:
    base: dict[str, object] = {
        "baseline_bytes": 7500 * MIB,
        "weights_bytes": 3000 * MIB,
        "expert": ExpertCost(bytes_per_slot=100 * MIB, min_slots=4, max_slots=64),
        "kv_reserve_tokens": 4096,
    }
    base.update(overrides)
    return _inputs(**base)


# -- the table: machine x model ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("baseline_mib", "weights_mib", "kv_blocks"),
    [
        pytest.param(3700, 2861, 1072, id="4GB-fp16"),
        pytest.param(5600, 2861, 4980, id="6GB-fp16"),
        pytest.param(7500, 2861, 8889, id="8GB-fp16"),
        pytest.param(11500, 2861, 17117, id="12GB-fp16"),
        pytest.param(3700, 1431, 4340, id="4GB-8bit-weights"),
    ],
)
def test_dense_table_fills_the_pool_with_kv(
    baseline_mib: int, weights_mib: int, kv_blocks: int
) -> None:
    plan = plan_budget(_inputs(baseline_bytes=baseline_mib * MIB, weights_bytes=weights_mib * MIB))

    assert plan.usable_bytes == baseline_mib * MIB * 9 // 10
    assert plan.headroom_bytes == baseline_mib * MIB - plan.usable_bytes
    assert plan.kv_blocks == (kv_blocks,)
    assert plan.kv_tokens == (kv_blocks - 1) * 16
    assert plan.expert_slots == 0
    assert plan.expert_bytes == 0
    assert plan.loaded_drafts == ()
    assert plan.skipped == ()
    assert plan.max_running == (kv_blocks - 1) // 64


def test_the_4gb_case_matches_the_strategy_example() -> None:
    plan = plan_budget(_inputs())

    # About 17,100 tokens of context on a 4 GB card with fp16 weights.
    assert plan.kv_tokens == 17136
    assert plan.kv_bytes == 1072 * BLOCK_BYTES
    assert plan.max_running == 16


def test_moe_case_fills_experts_before_kv() -> None:
    plan = plan_budget(_expert_inputs())

    # pool = 6750 - 3000 = 3750 MiB; the KV floor (257 blocks) is 112.43 MiB, so 36 slots
    # fit; the 150 MiB left over holds 342 blocks.
    assert plan.expert_slots == 36
    assert plan.expert_bytes == 36 * 100 * MIB
    assert plan.kv_blocks == (342,)
    assert plan.kv_tokens == 341 * 16
    assert plan.kv_tokens >= 4096


def test_moe_expert_cache_is_clamped_to_its_maximum() -> None:
    plan = plan_budget(_expert_inputs(baseline_bytes=20000 * MIB))

    assert plan.expert_slots == 64
    assert plan.kv_blocks[0] > 257


def test_two_groups_share_the_block_count_and_both_are_paid_for() -> None:
    sliding = KVCost(bytes_per_block=BLOCK_BYTES // 2, block_size=16)
    plan = plan_budget(_inputs(kv=(DENSE_KV, sliding)))

    pool = 3330 * MIB - WEIGHTS_FP16
    assert plan.kv_blocks == (pool // (BLOCK_BYTES + BLOCK_BYTES // 2),) * 2
    assert plan.kv_bytes <= pool


def test_fixed_kv_bytes_come_off_the_pool_before_blocks() -> None:
    state = KVCost(bytes_per_block=BLOCK_BYTES, fixed_bytes=100 * MIB, block_size=16)

    plan = plan_budget(_inputs(kv=(state,)))

    assert plan.kv_blocks == ((3330 * MIB - WEIGHTS_FP16 - 100 * MIB) // BLOCK_BYTES,)
    assert plan.kv_bytes == plan.kv_blocks[0] * BLOCK_BYTES + 100 * MIB


def test_late_allocation_is_charged_before_the_split() -> None:
    plain = plan_budget(_inputs())
    charged = plan_budget(_inputs(late_alloc_bytes=245 * MIB))

    assert charged.kv_blocks[0] == (3330 * MIB - WEIGHTS_FP16 - 245 * MIB) // BLOCK_BYTES
    assert charged.kv_tokens < plain.kv_tokens


def test_ratio_is_read_as_the_decimal_that_was_written() -> None:
    plan = plan_budget(
        _inputs(
            baseline_bytes=10,
            memory_ratio=0.7,
            weights_bytes=0,
            kv=(KVCost(1, block_size=1),),
            kv_reserve_tokens=1,
        )
    )

    assert plan.usable_bytes == 7
    assert plan.headroom_bytes == 3


# -- drafts -----------------------------------------------------------------------------------


def test_drafts_load_in_priority_order_and_the_rest_are_skipped_with_the_numbers() -> None:
    small = DraftCost("ngram", fixed_bytes=50 * MIB)
    huge = DraftCost("mtp", fixed_bytes=4000 * MIB)
    last = DraftCost("tiny", fixed_bytes=10 * MIB)

    plan = plan_budget(_inputs(drafts=(small, huge, last)))

    assert plan.loaded_drafts == ("ngram", "tiny")
    assert [name for name, _ in plan.skipped] == ["mtp"]
    assert "have" in plan.skipped[0][1]
    assert plan.draft_bytes == 60 * MIB
    assert plan.kv_blocks[0] == (3330 * MIB - WEIGHTS_FP16 - 60 * MIB) // BLOCK_BYTES


def test_draft_kv_is_charged_per_usable_token() -> None:
    draft = DraftCost("eagle", fixed_bytes=0, bytes_per_kv_token=1024)

    plan = plan_budget(_inputs(drafts=(draft,)))

    assert plan.loaded_drafts == ("eagle",)
    assert plan.draft_bytes == plan.kv_tokens * 1024
    used = WEIGHTS_FP16 + plan.kv_bytes + plan.draft_bytes
    assert used <= plan.usable_bytes
    # One more block would not have fit.
    assert plan.usable_bytes - used < BLOCK_BYTES + 16 * 1024


# -- fixed block count --------------------------------------------------------------------------


def test_an_override_fixes_the_block_count_and_spares_the_rest() -> None:
    plan = plan_budget(_inputs(kv_blocks_override=500))

    assert plan.kv_blocks == (500,)
    assert plan.kv_tokens == 499 * 16
    assert plan.kv_bytes == 500 * BLOCK_BYTES


def test_an_override_still_has_to_fit_the_usable_memory() -> None:
    with pytest.raises(CapacityError) as caught:
        plan_budget(_inputs(kv_blocks_override=5000))

    assert cast("int", caught.value.context["short"]) > 0


def test_an_override_below_the_floor_is_refused() -> None:
    with pytest.raises(CapacityError, match="fixed by the user"):
        plan_budget(_inputs(kv_blocks_override=10))


# -- headroom: chunk, running, graphs -----------------------------------------------------------


def test_chunk_limit_comes_from_the_headroom() -> None:
    plan = plan_budget(
        _inputs(
            activation_bytes_per_token=100 * 1024,
            activation_fixed_bytes=100 * MIB,
            safety_bytes=50 * MIB,
            graph_bytes_estimate=120 * MIB,
        )
    )

    # 370 MiB of headroom minus 270 MiB of fixed items leaves 100 MiB.
    assert plan.headroom_bytes == 370 * MIB
    assert plan.max_chunk_tokens == 100 * MIB // (100 * 1024)
    assert plan.graph_bytes == 120 * MIB


def test_unknown_activation_cost_leaves_the_chunk_bounded_by_kv() -> None:
    plan = plan_budget(_inputs())

    assert plan.max_chunk_tokens == plan.kv_tokens


def test_graphs_are_skipped_when_they_would_starve_the_chunk() -> None:
    plan = plan_budget(
        _inputs(activation_bytes_per_token=100 * 1024, graph_bytes_estimate=400 * MIB)
    )

    assert plan.graph_bytes == 0
    assert [name for name, _ in plan.skipped] == ["graph"]
    assert plan.max_chunk_tokens == 370 * MIB // (100 * 1024)


def test_no_room_for_one_prefill_block_is_refused() -> None:
    with pytest.raises(CapacityError, match="prefill block"):
        plan_budget(_inputs(activation_bytes_per_token=100 * MIB))


# -- refusals name three numbers ----------------------------------------------------------------


def test_weights_that_do_not_fit_report_need_have_and_short() -> None:
    with pytest.raises(CapacityError) as caught:
        plan_budget(_inputs(baseline_bytes=2000 * MIB))

    context = caught.value.context
    assert context["need"] == WEIGHTS_FP16
    assert context["have"] == 1800 * MIB
    assert context["short"] == WEIGHTS_FP16 - 1800 * MIB
    assert str(context["short"]) in str(caught.value)


def test_a_kv_floor_that_does_not_fit_reports_the_shortfall() -> None:
    with pytest.raises(CapacityError) as caught:
        plan_budget(_inputs(weights_bytes=3300 * MIB, kv_reserve_tokens=8192))

    context = caught.value.context
    assert context["what"] == "KV floor"
    assert context["need"] == 3300 * MIB + 513 * BLOCK_BYTES
    assert context["have"] == 3330 * MIB
    assert context["short"] == cast("int", context["need"]) - cast("int", context["have"])


def test_minimum_expert_slots_count_toward_the_floor() -> None:
    with pytest.raises(CapacityError):
        plan_budget(_expert_inputs(baseline_bytes=3700 * MIB))


# -- monotonicity -------------------------------------------------------------------------------

BASELINES = [mib * MIB for mib in range(3400, 24000, 137)]


def test_more_memory_never_shrinks_the_plan_for_a_dense_model() -> None:
    previous = None
    for baseline in BASELINES:
        plan = plan_budget(_inputs(baseline_bytes=baseline))
        if previous is not None:
            assert plan.kv_tokens >= previous.kv_tokens
            assert plan.usable_bytes >= previous.usable_bytes
            assert plan.headroom_bytes >= previous.headroom_bytes
            assert plan.max_running >= previous.max_running
        previous = plan


def test_more_memory_never_shrinks_the_expert_cache_or_drops_below_the_floor() -> None:
    previous = None
    for baseline in BASELINES[4:]:
        plan = plan_budget(_expert_inputs(baseline_bytes=baseline))
        assert plan.kv_tokens >= 4096
        if previous is not None:
            assert plan.expert_slots >= previous.expert_slots
        previous = plan


def test_more_memory_never_shrinks_kv_once_the_expert_cache_is_full() -> None:
    previous = None
    for baseline in range(20000, 30000, 211):
        plan = plan_budget(_expert_inputs(baseline_bytes=baseline * MIB))
        assert plan.expert_slots == 64
        if previous is not None:
            assert plan.kv_tokens >= previous.kv_tokens
        previous = plan


def test_a_heavier_model_never_grows_the_kv() -> None:
    previous = None
    for weights in range(500, 3300, 97):
        plan = plan_budget(_inputs(weights_bytes=weights * MIB))
        if previous is not None:
            assert plan.kv_tokens <= previous.kv_tokens
        previous = plan


def test_a_costlier_block_never_grows_the_kv() -> None:
    previous = None
    for per_block in range(BLOCK_BYTES // 2, BLOCK_BYTES * 2, 12345):
        plan = plan_budget(_inputs(kv=(KVCost(per_block, block_size=16),)))
        if previous is not None:
            assert plan.kv_tokens <= previous.kv_tokens
        previous = plan


def test_appending_a_draft_never_grows_the_kv() -> None:
    drafts = (
        DraftCost("a", 30 * MIB, 512),
        DraftCost("b", 900 * MIB),
        DraftCost("c", 5000 * MIB),
        DraftCost("d", 1 * MIB, 4096),
    )
    previous = plan_budget(_inputs())
    for count in range(1, len(drafts) + 1):
        plan = plan_budget(_inputs(drafts=drafts[:count]))
        assert plan.kv_tokens <= previous.kv_tokens
        previous = plan


def test_the_plan_never_spends_more_than_it_may_and_leaves_no_whole_block_unused() -> None:
    drafts = (DraftCost("eagle", 20 * MIB, 2048),)
    for baseline in BASELINES:
        inputs = _inputs(baseline_bytes=baseline, drafts=drafts)
        plan = plan_budget(inputs)
        used = WEIGHTS_FP16 + plan.kv_bytes + plan.draft_bytes + plan.expert_bytes
        assert used <= plan.usable_bytes
        assert plan.usable_bytes - used < BLOCK_BYTES + 16 * 2048


# -- input validation ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"memory_ratio": 0.0}, ValueError),
        ({"memory_ratio": 1.5}, ValueError),
        ({"memory_ratio": float("nan")}, ValueError),
        ({"memory_ratio": "0.9"}, TypeError),
        ({"baseline_bytes": -1}, ValueError),
        ({"weights_bytes": 1.5}, TypeError),
        ({"kv": ()}, ValueError),
        ({"kv": (KVCost(1, block_size=16), KVCost(1, block_size=32))}, ValueError),
        ({"kv_reserve_tokens": 0}, ValueError),
        ({"kv_blocks_override": 1}, ValueError),
    ],
)
def test_invalid_inputs_are_refused_at_construction(
    overrides: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        _inputs(**overrides)


def test_cost_records_validate_their_fields() -> None:
    with pytest.raises(ValueError, match="bytes_per_block"):
        KVCost(bytes_per_block=0)
    with pytest.raises(ValueError, match="max_slots"):
        ExpertCost(bytes_per_slot=1, min_slots=3, max_slots=2)
    with pytest.raises(ValueError, match="name"):
        DraftCost(" ", 0)


def test_the_plan_is_a_pure_function_of_its_inputs() -> None:
    inputs = _inputs(drafts=(DraftCost("a", MIB),))

    assert plan_budget(inputs) == plan_budget(replace(inputs))


@pytest.mark.parametrize("module", ["budget"])
def test_planning_does_not_import_torch(module: str) -> None:
    program = (
        "import sys\n"
        f"import ayaka.memory.{module}\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )

    result = subprocess.run(  # noqa: S603  # fixed argv, no input from outside
        [sys.executable, "-I", "-c", program], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr


# -- the ledger ---------------------------------------------------------------------------------


def test_reserve_and_release_move_bytes_between_owner_and_free() -> None:
    ledger = MemoryBudget(1000)

    ledger.reserve("weights", 600)
    ledger.reserve("kv", 300)
    ledger.release("kv", 100)

    assert ledger.held("weights") == 600
    assert ledger.held("kv") == 200
    assert ledger.reserved_bytes == 800
    assert ledger.free_bytes == 200
    assert ledger.capacity_bytes == 1000


def test_a_reservation_over_capacity_raises_and_changes_nothing() -> None:
    ledger = MemoryBudget(1000)
    ledger.reserve("weights", 900)
    before = ledger.snapshot()

    with pytest.raises(CapacityError) as caught:
        ledger.reserve("kv", 101)

    assert ledger.snapshot() == before
    assert caught.value.context == {"what": "reserve kv", "need": 101, "have": 100, "short": 1}


def test_a_reservation_of_exactly_the_free_bytes_succeeds() -> None:
    ledger = MemoryBudget(1000)

    ledger.reserve("kv", 1000)

    assert ledger.free_bytes == 0


def test_releasing_more_than_held_is_a_programming_error_and_changes_nothing() -> None:
    ledger = MemoryBudget(1000)
    ledger.reserve("graph", 50)

    with pytest.raises(ValueError, match="holds 50"):
        ledger.release("graph", 51)

    assert ledger.held("graph") == 50


def test_the_snapshot_lists_every_owner_and_is_a_copy() -> None:
    ledger = MemoryBudget(10)
    ledger.reserve("draft", 4)

    snapshot = dict(ledger.snapshot())
    snapshot["draft"] = 0

    assert set(snapshot) == set(BUDGET_OWNERS)
    assert ledger.held("draft") == 4
    assert ledger.snapshot()["weights"] == 0


@pytest.mark.parametrize("call", ["reserve", "release", "held"])
def test_an_unknown_owner_is_refused(call: str) -> None:
    ledger = MemoryBudget(10)
    args: tuple[object, ...] = ("nobody",) if call == "held" else ("nobody", 1)

    with pytest.raises(ValueError, match="unknown owner"):
        getattr(ledger, call)(*args)


@pytest.mark.parametrize("nbytes", [-1, 1.5, True])
def test_a_bad_byte_count_is_refused(nbytes: object) -> None:
    ledger = MemoryBudget(10)

    with pytest.raises((TypeError, ValueError)):
        ledger.reserve("kv", nbytes)  # type: ignore[arg-type]  # the point of the test


def test_a_plan_reserves_cleanly_in_a_ledger_sized_by_its_usable_bytes() -> None:
    plan = plan_budget(_expert_inputs(drafts=(DraftCost("ngram", 8 * MIB),)))
    ledger = MemoryBudget(plan.usable_bytes)

    ledger.reserve("weights", 3000 * MIB)
    ledger.reserve("kv", plan.kv_bytes)
    ledger.reserve("draft", plan.draft_bytes)
    ledger.reserve("expert_cache", plan.expert_bytes)

    assert ledger.free_bytes >= 0
    assert ledger.free_bytes < BLOCK_BYTES + 100 * MIB
