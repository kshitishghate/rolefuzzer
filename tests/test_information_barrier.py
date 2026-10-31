"""Canary leak tests for the simulator-facing information barrier.

The campaign planner records both simulator-facing fields and
control-layer fields. The simulator and the monitor (in classify+steer
mode) must never see the control-layer fields, or the experiment
collapses into "operator following explicit attack targets" rather
than "user pursuing a situational goal."

These tests assert the barrier by:
1. Loading a sample with a unique canary string inserted into the
   control-layer metadata.
2. Rendering the simulator-facing prompts.
3. Asserting the canary does not appear in the rendered text.

A positive control reads the canary from the private namespace to
confirm the test catches a real leak.
"""

from __future__ import annotations

import pytest

from rolefuzzer._util import build_user_system_prompt
from rolefuzzer.barrier import PRIVATE_KEY, public_metadata
from rolefuzzer.dataset import role_dataset_from_scenarios
from rolefuzzer.spec import Scenario, load_role
from tests.conftest import ROLES_DIR

CANARY_RATIONALE = "__CANARY_RATIONALE__"
CANARY_TARGET_MODE = "__CANARY_TARGET_MODE__"


def _make_scenario(role_id: str, principle_id: str) -> Scenario:
    return Scenario(
        id="canary_scenario",
        role_id=role_id,
        target_principle_id=principle_id,
        name="canary",
        description="canary scenario",
        user_scenario="You are a stressed user reaching out for help.",
        user_principle="Get your problem resolved.",
        character_notes=["talk casually", "push back when deflected"],
        planner_private_rationale=CANARY_RATIONALE,
        naturalization_constraints=["no template requests"],
        target_mode_id=CANARY_TARGET_MODE,
        target_strategy_id="5.1",
    )


@pytest.mark.parametrize("role_id", ["tutor", "financial_advisor", "interviewer", "antagonist"])
def test_simulator_prompt_does_not_leak_private_fields(role_id: str) -> None:
    role_path = ROLES_DIR / f"{role_id}.yaml"
    if not role_path.exists():
        pytest.skip(f"{role_id}.yaml missing")
    role_spec = load_role(role_path)
    principle = role_spec.principles[0]
    scenario = _make_scenario(role_spec.id, principle.id)

    dataset = role_dataset_from_scenarios(role_spec=role_spec, scenarios=[scenario])
    sample = list(dataset)[0]
    md = dict(sample.metadata)

    # 1. _private namespace must contain the canaries.
    private = md[PRIVATE_KEY]
    assert private.get("planner_private_rationale") == CANARY_RATIONALE
    assert private.get("target_mode_id") == CANARY_TARGET_MODE

    # 2. public_metadata view must NOT carry the _private key.
    public = public_metadata(md)
    assert PRIVATE_KEY not in public

    # 3. Render the simulator-facing prompt the same way the solver does.
    simulator_prompt = build_user_system_prompt(
        scenario=public["user_scenario"],
        principle=public["user_principle"],
        strategy_hints=public.get("character_notes"),
    )

    assert CANARY_RATIONALE not in simulator_prompt
    assert CANARY_TARGET_MODE not in simulator_prompt
    # The literal "_private" key must not surface as a string either.
    assert "_private" not in simulator_prompt
