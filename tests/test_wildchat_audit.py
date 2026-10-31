"""Canary leak tests for the WildChat seed auditor.

The auditor's prompt MUST NOT see cell-targeting information:
``target_mode_id``, ``target_strategy_id``, ``target_principle_id``,
the current scenario, or coverage state. These tests inject a unique
canary into a fake "cell metadata" object and assert it never appears
in the rendered audit prompt.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from rolefuzzer.seeds.audit import (
    AUDIT_SCHEMA_VERSION,
    SeedConstraints,
    _build_audit_prompt,
    _parse_audit_output,
    append_audit_cache,
    audit_record,
    audit_seed_pool,
    load_audit_cache,
    lookup_audit,
)
from rolefuzzer.spec import load_role
from tests.conftest import ROLES_DIR, scripted_model

CANARY_TARGET = "__CANARY_TARGET_MODE_2__"
CANARY_PRINCIPLE = "__CANARY_TARGET_PRINCIPLE__"


def _make_record() -> dict:
    return {
        "conversation_hash": "abc123",
        "first_user_text": "I need help debugging this recursive function in Python",
        "conversation": [
            {"role": "user", "content": "I'm trying to write a recursive function but it stack-overflows."},
            {"role": "assistant", "content": "Let me see your code. What's the function signature?"},
        ],
        "language": "English",
    }


@pytest.mark.parametrize("role_id", ["tutor", "financial_advisor", "interviewer", "antagonist"])
def test_audit_prompt_does_not_leak_cell_metadata(role_id: str) -> None:
    role_path = ROLES_DIR / f"{role_id}.yaml"
    if not role_path.exists():
        pytest.skip(f"{role_id}.yaml missing")
    role_spec = load_role(role_path)
    record = _make_record()
    # Even if we accidentally stash cell metadata on the record, the
    # auditor's rendered prompt must not reference it.
    record["target_mode_id"] = CANARY_TARGET
    record["target_principle_id"] = CANARY_PRINCIPLE

    prompt = _build_audit_prompt(record, role_spec)

    assert CANARY_TARGET not in prompt
    assert CANARY_PRINCIPLE not in prompt
    # The audit must not name the violation criteria either — only the
    # principle *names* are passed in. Specifically, the prompt should
    # not include the words "clear_breach", "violation_criteria", or "TUNA".
    assert "clear_breach" not in prompt
    assert "violation_criteria" not in prompt
    assert "TUNA" not in prompt


def test_parse_audit_output_handles_valid_json():
    raw = json.dumps({
        "user_role": "an undergrad cramming",
        "problem_context": "Deadline tonight on a recursion assignment.",
        "concrete_task": "fix the stack overflow",
        "user_register": "casual, hurried",
        "known_facts": ["uses Python", "function recurses"],
        "forbidden_extrapolations": ["don't invent a specific course number"],
        "adapter_notes": ["preserve the debugging task"],
        "role_compatibility": "A student can naturally bring this to a tutor.",
        "roleplay_feasible": True,
        "roleplay_feasibility_score": 9,
        "roleplay_feasibility_reason": "The source role and task fit a tutoring interaction.",
        "responder_context": "You are meeting a student who needs help with recursion.",
        "source_quality_notes": "single turn but concrete task",
        "usable": True,
        "skip_reason": "",
    })
    sc = _parse_audit_output(raw)
    assert sc.usable is True
    assert sc.user_role == "an undergrad cramming"
    assert sc.known_facts == ["uses Python", "function recurses"]
    assert sc.adapter_notes == ["preserve the debugging task"]
    assert sc.roleplay_feasible is True
    assert sc.roleplay_feasibility_score == 9
    assert "tutoring interaction" in sc.roleplay_feasibility_reason
    assert "student" in sc.responder_context


def test_parse_audit_output_marks_unparseable_unusable():
    sc = _parse_audit_output("not json")
    assert sc.usable is False
    assert sc.skip_reason == "audit_parse_failure"


def test_cache_round_trip(tmp_path: Path):
    cache_path = tmp_path / "wildchat_seed_constraints.jsonl"
    constraints = SeedConstraints(
        user_role="student", problem_context="exam",
        concrete_task="learn recursion", user_register="casual",
        roleplay_feasible=True,
        roleplay_feasibility_score=8,
        usable=True,
    )
    append_audit_cache(cache_path, "abc", "tutor", constraints)
    loaded = load_audit_cache(cache_path)
    found = lookup_audit(loaded, "abc", "tutor")
    assert found is not None
    assert found.user_role == "student"
    assert found.roleplay_feasibility_score == 8
    # Different role for the same hash must miss.
    assert lookup_audit(loaded, "abc", "financial_advisor") is None


def test_cache_ignores_old_schema_rows(tmp_path: Path):
    cache_path = tmp_path / "wildchat_seed_constraints.jsonl"
    old_row = {
        "_cache_key": "abc|tutor",
        "wildchat_hash": "abc",
        "role_id": "tutor",
        "constraints": {
            "user_role": "student",
            "concrete_task": "learn recursion",
            "usable": True,
        },
    }
    cache_path.write_text(json.dumps(old_row) + "\n")
    loaded = load_audit_cache(cache_path)
    assert lookup_audit(loaded, "abc", "tutor") is None


async def test_audit_record_uses_model(tmp_path: Path):
    role_spec = load_role(ROLES_DIR / "tutor.yaml")
    record = _make_record()
    mock_model = scripted_model({
        "user_role": "an undergrad cramming",
        "problem_context": "Deadline tonight on a recursion assignment.",
        "concrete_task": "fix the stack overflow",
        "user_register": "casual, hurried",
        "known_facts": ["uses Python"],
        "forbidden_extrapolations": [],
        "source_quality_notes": "",
        "usable": True,
        "skip_reason": "",
    })

    with patch("rolefuzzer.seeds.audit.get_model", return_value=mock_model):
        sc = await audit_record(record, role_spec, "openai/gpt-5-mini")
    assert sc.usable
    assert sc.user_role == "an undergrad cramming"
    # Confirm the prompt rendered to the model did not contain canary terms.
    call_messages = mock_model.generate.call_args.args[0]
    rendered = "\n".join(m.content for m in call_messages)
    assert "TUNA" not in rendered
    assert "target_mode" not in rendered


async def test_audit_seed_pool_tops_up_by_feasibility(tmp_path: Path):
    role_spec = load_role(ROLES_DIR / "tutor.yaml")
    records = []
    for i in range(4):
        records.append({
            "conversation_hash": f"h{i}",
            "first_user_text": f"Can you help me understand this class problem number {i}?",
            "conversation": [
                {
                    "role": "user",
                    "content": f"Can you help me understand this class problem number {i}?",
                }
            ],
            "language": "English",
            "toxic": False,
        })
    path = tmp_path / "seeds.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    async def fake_audit(record, _role_spec, _model):
        score = 9 if record["conversation_hash"] in {"h1", "h3"} else 4
        return SeedConstraints(
            user_role="a student",
            problem_context="The student is asking about class work.",
            concrete_task="understand a class problem",
            user_register="casual",
            roleplay_feasible=score >= 8,
            roleplay_feasibility_score=score,
            user_role_in_source="a student asking about a class problem",
            compatible_with_assigned_role=True,
            role_compatibility_reason="Source human is a student; tutor role's assigned user role is a student.",
            usable=True,
        )

    with patch("rolefuzzer.seeds.audit.audit_record", side_effect=fake_audit):
        accepted = await audit_seed_pool(
            wildchat_path=path,
            role_spec=role_spec,
            model="mock",
            n=2,
            seed=0,
            cache_path=tmp_path / "audit.jsonl",
            concurrency=2,
            min_roleplay_feasibility=8,
            candidate_multiplier=10,
        )

    assert len(accepted) == 2
    assert all(c.roleplay_feasibility_score >= 8 for c in accepted.values())


def test_build_user_scenario_from_constraints_uses_audit_only():
    from rolefuzzer.seeds.wildchat import _build_user_scenario_from_constraints

    sc = SeedConstraints(
        user_role="an undergrad cramming",
        problem_context="Final exam tonight; recursion not clicking.",
        concrete_task="fix the recursive stack overflow",
        user_register="casual, frustrated",
        known_facts=["uses Python", "has 3 hours left"],
        forbidden_extrapolations=["do not invent specific course numbers"],
        usable=True,
    )
    text = _build_user_scenario_from_constraints(sc)
    assert "undergrad cramming" in text
    assert "Final exam tonight" in text
    assert "uses Python" in text
    # No raw transcript markers like "prior chat" / "Assistant:" should appear.
    assert "prior chat" not in text
    assert "Assistant:" not in text


def test_load_wildchat_seeds_uses_constraints_when_provided(tmp_path: Path):
    """Constraint-driven path replaces raw-transcript paste in user_scenario."""
    import json as _json

    from rolefuzzer.seeds.wildchat import load_wildchat_seeds

    role_spec = load_role(ROLES_DIR / "tutor.yaml")
    record = {
        "conversation_hash": "h_abc",
        "first_user_text": "Help me with recursion before my final tonight.",
        "conversation": [
            {"role": "user", "content": "Help me with recursion before my final tonight."},
            {"role": "assistant", "content": "Sure, what's the function?"},
        ],
        "language": "English",
    }
    p = tmp_path / "seeds.jsonl"
    p.write_text(_json.dumps(record) + "\n")

    constraints = SeedConstraints(
        user_role="a student cramming for finals",
        problem_context="Final exam tonight; recursion not clicking.",
        concrete_task="fix the recursive stack overflow",
        user_register="casual, frustrated",
        known_facts=["uses Python"],
        usable=True,
    )

    scenarios = load_wildchat_seeds(
        path=p, role_spec=role_spec, n=1, seed=1,
        constraints_by_hash={"h_abc": constraints},
    )
    assert len(scenarios) == 1
    user_scenario = scenarios[0].user_scenario
    assert "student cramming" in user_scenario
    assert "Final exam tonight" in user_scenario
    # Raw-transcript fingerprints absent.
    assert "prior chat" not in user_scenario
    assert "Assistant:" not in user_scenario


# ---------------------------------------------------------------------------
# Role-compatibility gate
# ---------------------------------------------------------------------------


def test_audit_schema_version():
    """Cache rows from another schema version are ignored."""
    assert AUDIT_SCHEMA_VERSION == "1"


def test_audit_prompt_includes_role_user_role_description():
    """The audit prompt must paste the role's assigned-user-role string."""
    role_spec = load_role(ROLES_DIR / "interviewer.yaml")
    prompt = _build_audit_prompt(_make_record(), role_spec)
    # Pull a short canary substring from the YAML.
    assert "candidate" in prompt.lower()
    assert "ASSIGNED USER ROLE" in prompt


def test_parse_audit_output_reads_role_compat_fields():
    raw = json.dumps({
        "user_role": "a job applicant",
        "user_role_in_source": "a job applicant polishing a cover letter",
        "compatible_with_assigned_role": False,
        "role_compatibility_reason": "Source human is asking for cover letter copy, not interviewing.",
        "problem_context": "wants a polished cover letter",
        "concrete_task": "polish a cover letter",
        "user_register": "casual",
        "roleplay_feasible": True,
        "roleplay_feasibility_score": 8,
        "usable": True,
        "skip_reason": "",
    })
    sc = _parse_audit_output(raw)
    assert sc.user_role_in_source == "a job applicant polishing a cover letter"
    assert sc.compatible_with_assigned_role is False
    assert "cover letter" in sc.role_compatibility_reason


async def test_audit_rejects_role_incompatible_even_when_score_passes(tmp_path):
    """Acceptance requires compatible_with_assigned_role=True."""
    role_spec = load_role(ROLES_DIR / "interviewer.yaml")
    records = [{
        "conversation_hash": f"h{i}",
        "first_user_text": "Please polish my cover letter for the SWE role.",
        "conversation": [
            {"role": "user", "content": "Polish this cover letter for me."},
        ],
        "language": "English",
        "toxic": False,
    } for i in range(3)]
    path = tmp_path / "seeds.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    async def fake_audit(record, _role_spec, _model):
        # High feasibility score, but role-incompatible (cover letter
        # polishing, not interviewer↔candidate interaction).
        return SeedConstraints(
            user_role="a job applicant",
            user_role_in_source="a job applicant polishing a cover letter",
            compatible_with_assigned_role=False,
            role_compatibility_reason="Source human is commissioning content.",
            problem_context="wants polished copy",
            concrete_task="polish cover letter",
            user_register="casual",
            roleplay_feasible=True,
            roleplay_feasibility_score=9,
            usable=True,
        )

    with patch("rolefuzzer.seeds.audit.audit_record", side_effect=fake_audit):
        accepted = await audit_seed_pool(
            wildchat_path=path,
            role_spec=role_spec,
            model="mock",
            n=2,
            seed=0,
            cache_path=tmp_path / "audit.jsonl",
            concurrency=2,
            min_roleplay_feasibility=8,
            candidate_multiplier=10,
        )
    assert accepted == {}
