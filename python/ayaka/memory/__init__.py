"""Where a device's memory goes: the budget ledger, the plan that fills it, the step workspace.

``plan_budget`` is a pure function of integers, so planning needs no device and no
torch; ``MemoryBudget`` is the ledger the engine reserves the result in;
``Workspace`` is the scratch buffer a step takes temporary memory from. Public
names are resolved from their submodule on first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.memory.budget import (
        BUDGET_OWNERS,
        BudgetInputs,
        BudgetOwner,
        BudgetPlan,
        DraftCost,
        ExpertCost,
        KVCost,
        MemoryBudget,
        plan_budget,
    )
    from ayaka.memory.workspace import WORKSPACE_ALIGNMENT, Workspace

__all__ = [
    "BUDGET_OWNERS",
    "WORKSPACE_ALIGNMENT",
    "BudgetInputs",
    "BudgetOwner",
    "BudgetPlan",
    "DraftCost",
    "ExpertCost",
    "KVCost",
    "MemoryBudget",
    "Workspace",
    "plan_budget",
]

_EXPORTS = {
    "budget": (
        "BUDGET_OWNERS",
        "BudgetInputs",
        "BudgetOwner",
        "BudgetPlan",
        "DraftCost",
        "ExpertCost",
        "KVCost",
        "MemoryBudget",
        "plan_budget",
    ),
    "workspace": ("WORKSPACE_ALIGNMENT", "Workspace"),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
