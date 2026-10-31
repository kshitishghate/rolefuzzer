"""Scenario generation for a target cell and for the no-grid baseline."""

from __future__ import annotations

import logging

from inspect_ai.model import Model, get_model

from rolefuzzer._util import load_prompt
from rolefuzzer.planner.scenarios import (
    _assert_responder_context_required,
    _execution_constraints,
    _generate_validated,
    _inherit_execution_context,
    _parse_scenario_output,
    _principle_prompt_fields,
)
from rolefuzzer.spec import (
    Principle,
    RoleSpec,
    Scenario,
)
from rolefuzzer.taxonomy import TunaMode, TunaStrategy, TunaTaxonomy

logger = logging.getLogger(__name__)


async def generate_scenario_for_cell(
    planner_model: str | Model,
    role_spec: RoleSpec,
    target_principle: Principle,
    target_mode: TunaMode,
    target_strategy: TunaStrategy,
    taxonomy: TunaTaxonomy,
    seed_scenario: Scenario | None = None,
    coverage_context: str = "",
    diagnostic_context: str = "",
) -> Scenario:
    """Generate a new Scenario targeting a (mode, strategy, principle) cell.

    Args:
        planner_model: Model for scenario generation.
        role_spec: The role spec being tested.
        target_principle: The principle to target.
        target_mode: The TUNA mode (1-4) to target.
        target_strategy: The TUNA strategy (5-6) to target.
        taxonomy: Full TUNA taxonomy (for context).
        seed_scenario: Optional seed scenario as inspiration.
        coverage_context: Pre-rendered coverage context text.
        diagnostic_context: Pre-rendered diagnostic context from prior runs.

    Returns:
        A new Scenario object with targeting fields set.
    """
    model = get_model(planner_model) if isinstance(planner_model, str) else planner_model

    seed_section = ""
    if seed_scenario:
        seed_section = (
            f"SEED SCENARIO (as inspiration):\n"
            f"  Name: {seed_scenario.name}\n"
            f"  User scenario: {seed_scenario.user_scenario}\n"
            f"  User goal: {seed_scenario.user_principle}\n"
        )

    seed_section += _execution_constraints(seed_scenario)

    prompt = load_prompt("planner_cell_generation.txt").format(
        target_mode_id=target_mode.id,
        target_mode_name=target_mode.name,
        target_strategy_id=target_strategy.id,
        target_strategy_name=target_strategy.name,
        seed_section=seed_section,
        coverage_context=coverage_context,
        diagnostic_context=diagnostic_context,
        tuna_modes_summary=taxonomy.render_modes_summary(),
        **_principle_prompt_fields(role_spec, target_principle),
    )

    def finish(completion: str) -> Scenario:
        scenario = _parse_scenario_output(
            completion, role_spec.id, target_principle.id, role_spec=role_spec,
        )
        _assert_responder_context_required(
            role_spec, scenario.responder_context, scenario.user_scenario,
        )
        scenario = _inherit_execution_context(scenario, seed_scenario)
        return scenario.model_copy(update={
            "target_mode_id": target_mode.id,
            "target_strategy_id": target_strategy.id,
        })

    return await _generate_validated(
        model, prompt, finish,
        label=f"cell ({target_mode.id}, {target_strategy.id}, {target_principle.id})",
    )


async def generate_scenario_no_grid(
    planner_model: str | Model,
    role_spec: RoleSpec,
    target_principle: Principle,
    execution_anchor: Scenario | None = None,
) -> Scenario:
    """Generate a stress scenario from the role spec alone (no grid).

    The prompt-only generator behind the ``no-grid`` search mode. It
    receives the same role specification the coverage planner sees (role,
    principles, the target principle's pressure contract and clear-breach
    pattern) but no TUNA cell target, no coverage context, no diagnostics,
    and no seed anchor. This isolates the value of the grid apparatus
    (coverage-directed targeting + mutation + seed grounding) from simply
    asking a capable LLM to design targeted pressure.

    Role-general: the only role-specific inputs are the schema fields
    (role description, allowed modalities, pressure contract, invariants).

    ``execution_anchor`` is for roles whose scenarios execute in an existing
    environment: it supplies only the environment facts (the task and the
    role's specification for that environment, via the same block the cell
    generator uses) and binds the scenario to that environment. It adds no
    cell target, coverage context, or seed conversation.
    """
    model = get_model(planner_model) if isinstance(planner_model, str) else planner_model

    prompt = load_prompt("planner_no_grid.txt").format(
        **_principle_prompt_fields(role_spec, target_principle),
    )
    prompt += _execution_constraints(execution_anchor)

    def finish(completion: str) -> Scenario:
        scenario = _parse_scenario_output(
            completion, role_spec.id, target_principle.id, role_spec=role_spec,
        )
        _assert_responder_context_required(
            role_spec, scenario.responder_context, scenario.user_scenario,
        )
        scenario = _inherit_execution_context(scenario, execution_anchor)
        # No seed frame anchors a no-grid scenario (the execution anchor
        # supplies environment facts only).
        metadata = {**scenario.metadata, "arm": "no_grid", "frame_anchor_seed_id": ""}
        if execution_anchor is not None:
            metadata["execution_anchor_id"] = execution_anchor.id
            metadata["execution_anchor_principle_id"] = (
                execution_anchor.target_principle_id
            )
        return scenario.model_copy(update={
            "target_mode_id": "",
            "target_strategy_id": "",
            "metadata": metadata,
        })

    return await _generate_validated(
        model, prompt, finish, label=f"no-grid principle {target_principle.id}",
    )
