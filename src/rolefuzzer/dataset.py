"""Dataset construction for rolefuzzer.

Loads role YAML specs and scenario YAML files, cross-validates references,
and builds Inspect MemoryDataset objects with properly structured Samples.
"""

from __future__ import annotations

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem

from rolefuzzer.spec import (
    RoleSpec,
    Scenario,
    render_all_violation_criteria,
    render_violation_criteria,
)


def role_dataset_from_scenarios(
    role_spec: RoleSpec,
    scenarios: list[Scenario],
    iterations: int = 1,
) -> MemoryDataset:
    """Build an Inspect dataset from in-memory Scenario objects.

    Same Sample construction logic as ``role_dataset()`` but skips
    YAML file loading. Used by the campaign planner for dynamically
    generated scenarios.

    Args:
        role: The RoleSpec to test against.
        scenarios: List of in-memory Scenario objects.
        iterations: Number of iterations per scenario.

    Returns:
        MemoryDataset with one Sample per (scenario, iteration) pair.
    """
    samples: list[Sample] = []
    all_violation_criteria_text = render_all_violation_criteria(role_spec.principles)
    all_principle_ids = [p.id for p in role_spec.principles]
    id_counts: dict[str, int] = {}

    for scenario in scenarios:
        _validate_scenario(role_spec, scenario)

        target_principle = role_spec.get_principle(scenario.target_principle_id)
        violation_criteria_text = render_violation_criteria(target_principle)

        for iteration in range(iterations):
            base_id = f"{scenario.id}_iter{iteration}"
            count = id_counts.get(base_id, 0)
            id_counts[base_id] = count + 1
            sample_id = base_id if count == 0 else f"{base_id}_r{count}"
            sample = Sample(
                input=[
                    ChatMessageSystem(
                        content=role_spec.render_system_prompt(
                            responder_context=scenario.responder_context,
                        )
                    ),
                ],
                target=violation_criteria_text,
                id=sample_id,
                metadata={
                    "role_id": role_spec.id,
                    "role_name": role_spec.name,
                    "scenario_id": scenario.id,
                    "scenario_name": scenario.name,
                    "target_principle_id": target_principle.id,
                    "target_principle_name": target_principle.name,
                    "violation_criteria_text": violation_criteria_text,
                    "all_violation_criteria_text": all_violation_criteria_text,
                    "all_principle_ids": all_principle_ids,
                    "user_scenario": scenario.user_scenario,
                    "user_principle": scenario.user_principle,
                    "character_notes": scenario.character_notes,
                    "pressure_strategy": scenario.pressure_strategy,
                    "parent_scenario_id": scenario.parent_scenario_id,
                    "mutation_type": scenario.mutation_type,
                    "responder_starts": role_spec.responder_starts,
                    "iteration": iteration,
                    "user_role": scenario.user_role,
                    "interaction_modality": scenario.interaction_modality,
                    "user_role_description": role_spec.user_role_description,
                    "allowed_interaction_modalities": list(
                        role_spec.allowed_interaction_modalities
                    ),
                    # control-layer fields live under metadata["_private"]
                    # so ``public_metadata()`` strips them at every simulator
                    # render boundary.
                    "_private": {
                        "target_mode_id": scenario.target_mode_id,
                        "target_strategy_id": scenario.target_strategy_id,
                        "planner_private_rationale": scenario.planner_private_rationale,
                        "pressure_self_check": scenario.pressure_self_check,
                        "naturalization_constraints": list(
                            scenario.naturalization_constraints
                        ),
                    },
                    # ``target_mode_id`` / ``target_strategy_id`` are also
                    # required by the solver's steering loop and the
                    # campaign-level grid updater (both planner-side
                    # consumers, never reach the simulator). They stay at
                    # the top level because planner-side code (e.g.
                    # ``assess_round``) reads them there; the
                    # information barrier is enforced by ``public_metadata``
                    # at the simulator render boundary, not by hiding the
                    # IDs from planner-side consumers.
                    "target_mode_id": scenario.target_mode_id,
                    "target_strategy_id": scenario.target_strategy_id,
                    **scenario.metadata,
                },
            )
            samples.append(sample)

    return MemoryDataset(samples)


def _validate_scenario(role_spec: RoleSpec, scenario: Scenario) -> None:
    """Validate that a scenario's cross-references are correct."""
    if scenario.role_id != role_spec.id:
        raise ValueError(
            f"Scenario '{scenario.id}' references role '{scenario.role_id}', "
            f"but was loaded with role '{role_spec.id}'."
        )
    # This will raise ValueError if the principle doesn't exist
    role_spec.get_principle(scenario.target_principle_id)
