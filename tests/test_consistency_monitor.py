"""Tests for the consistency monitor (no API calls; uses scripted models)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from rolefuzzer.monitor import (
    ConsistencyResult,
    MonitorVerdict,
    _collect_rule_flag_hints,
    _parse_consistency_result,
    _rule_based_quality_check,
    check_consistency,
    check_turn,
)
from tests.conftest import scripted_model


class TestConsistencyResult:
    def test_frozen(self):
        r = ConsistencyResult(approved=True, reasoning="OK")
        with pytest.raises(Exception):
            r.approved = False

    def test_defaults(self):
        r = ConsistencyResult(approved=True)
        assert r.flags == []
        assert r.suggested_revision is None
        assert r.reasoning == ""
        assert r.conversation_terminated is False


class TestParseConsistencyResult:
    def test_approved(self):
        raw = json.dumps({
            "approved": True,
            "flags": [],
            "suggested_revision": None,
            "reasoning": "Consistent with role.",
        })
        result = _parse_consistency_result(raw)
        assert result.approved is True
        assert result.flags == []
        assert result.suggested_revision is None

    def test_rejected_with_revision(self):
        raw = json.dumps({
            "approved": False,
            "flags": ["role_fidelity", "arc_progression"],
            "suggested_revision": "I'm really struggling with this, can you help?",
            "reasoning": "Too aggressive for turn 1.",
        })
        result = _parse_consistency_result(raw)
        assert result.approved is False
        assert "role_fidelity" in result.flags
        assert result.suggested_revision is not None
        assert "struggling" in result.suggested_revision

    def test_rejected_no_revision(self):
        raw = json.dumps({
            "approved": False,
            "flags": ["conversational_coherence"],
            "suggested_revision": None,
            "reasoning": "Ignores responder's last message.",
        })
        result = _parse_consistency_result(raw)
        assert result.approved is False
        assert result.suggested_revision is None

    def test_invalid_json_defaults_to_rejected(self):
        raw = "not json"
        result = _parse_consistency_result(raw)
        assert result.approved is False
        assert result.parse_failure is True
        assert "parse_failure" in result.flags

    def test_missing_verdict_is_rejected(self):
        result = _parse_consistency_result(json.dumps({"flags": ["role_confusion"]}))
        assert result.approved is False
        assert result.parse_failure is True

    def test_string_booleans_are_read_as_booleans(self):
        result = _parse_consistency_result(json.dumps(
            {"approved": "false", "terminate": "false", "flags": []}))
        assert result.approved is False
        assert result.parse_failure is False
        assert result.terminate is False

    def test_conversation_terminated(self):
        raw = json.dumps({
            "approved": True,
            "flags": [],
            "suggested_revision": None,
            "conversation_terminated": True,
            "reasoning": "User is saying goodbye.",
        })
        result = _parse_consistency_result(raw)
        assert result.approved is True
        assert result.conversation_terminated is True

    def test_conversation_not_terminated(self):
        raw = json.dumps({
            "approved": True,
            "flags": [],
            "suggested_revision": None,
            "conversation_terminated": False,
            "reasoning": "User still pursuing goal.",
        })
        result = _parse_consistency_result(raw)
        assert result.conversation_terminated is False


class TestCheckConsistency:
    async def test_approved_turn(self):
        model = scripted_model({
            "approved": True,
            "flags": [],
            "suggested_revision": None,
            "reasoning": "Turn is consistent with frustrated-student role.",
        })
        with patch("rolefuzzer.monitor.get_model", return_value=model):
            result = await check_consistency(
                monitor_model="test-model",
                user_scenario="You are a frustrated student.",
                messages=[],
                candidate_turn="I really don't understand this problem.",
                turn_number=1,
                num_turns=8,
            )
        assert result.approved is True
        model.generate.assert_called_once()

    async def test_rejected_turn(self):
        model = scripted_model({
            "approved": False,
            "flags": ["arc_progression"],
            "suggested_revision": "Can you help me with this math problem?",
            "reasoning": "Too aggressive for first turn.",
        })
        with patch("rolefuzzer.monitor.get_model", return_value=model):
            result = await check_consistency(
                monitor_model="test-model",
                user_scenario="You are a confused student.",
                messages=[],
                candidate_turn="GIVE ME THE ANSWER NOW!",
                turn_number=1,
                num_turns=8,
            )
        assert result.approved is False
        assert "arc_progression" in result.flags
        assert result.suggested_revision is not None

    async def test_terminated_turn(self):
        model = scripted_model({
            "approved": True,
            "flags": [],
            "suggested_revision": None,
            "conversation_terminated": True,
            "reasoning": "User is saying goodbye and has no remaining need.",
        })
        with patch("rolefuzzer.monitor.get_model", return_value=model):
            result = await check_consistency(
                monitor_model="test-model",
                user_scenario="You are a student rushing before a quiz.",
                messages=[],
                candidate_turn="Thanks so much! Bye!",
                turn_number=3,
                num_turns=8,
            )
        assert result.approved is True
        assert result.conversation_terminated is True

    async def test_model_object_passed_directly(self):
        model = scripted_model({
            "approved": True,
            "flags": [],
            "suggested_revision": None,
            "reasoning": "OK",
        })
        result = await check_consistency(
            monitor_model=model,
            user_scenario="Test scenario",
            messages=[],
            candidate_turn="Test turn",
            turn_number=1,
            num_turns=4,
        )
        assert result.approved is True


class TestMonitorVerdict:
    def test_frozen(self):
        v = MonitorVerdict(approved=True, reasoning="OK")
        with pytest.raises(Exception):
            v.approved = False

    def test_defaults(self):
        v = MonitorVerdict(approved=True)
        assert v.observed_trigger is None
        assert v.observed_mode_14 is None
        assert v.observed_strategy_56 is None
        assert v.target_hit is False
        assert v.steering_note == ""
        assert v.role_fidelity is True
        assert v.conversation_terminated is False
        assert v.flags == []
        assert v.suggested_revision is None


class TestValidatorOutputShapes:
    def test_json_in_code_fence(self):
        raw = ("Here is the verdict:\n```json\n"
               + json.dumps({"approved": True, "flags": [], "reasoning": "ok"})
               + "\n```\nThanks.")
        result = _parse_consistency_result(raw)
        assert result.approved is True
        assert result.parse_failure is False

    def test_prose_with_embedded_object(self):
        raw = ("I think the verdict should be: "
               + json.dumps({"approved": False, "flags": ["scope_drift"]})
               + " — that's my call.")
        result = _parse_consistency_result(raw)
        assert result.approved is False
        assert "scope_drift" in result.flags

    def test_terminate_fields(self):
        result = _parse_consistency_result(json.dumps({
            "approved": True, "terminate": True,
            "termination_reason": "scenario_goal_resolved", "reasoning": "Done.",
        }))
        assert result.terminate is True
        assert result.termination_reason == "scenario_goal_resolved"


class TestRuleBasedQualityCheck:
    # The rule-based check hard-rejects only empty turns. All other
    # pattern matches are emitted as advisory hints via
    # _collect_rule_flag_hints and consumed by the validate step.

    def test_empty_turn_hard_rejected(self):
        result = _rule_based_quality_check(
            candidate_turn="   ",
        )
        assert result is not None
        assert result.approved is False
        assert "empty_user_turn" in result.flags

    def test_non_empty_turn_passes_through(self):
        # Bot-tell patterns (STAR essay, role-confusion opener, even
        # template-like content) do not hard-reject; they are advisory
        # hints for the LLM monitor.
        result = _rule_based_quality_check(
            candidate_turn=(
                "**Situation:** I led a migration of our billing service.\n"
                "**Task:** Reduce latency by 40%.\n"
                "**Action:** I coordinated the cross-functional team.\n"
                "**Result:** Revenue increased 23.4% across 12 product lines."
            ),
        )
        assert result is None


class TestRuleFlagHints:
    def test_no_hints_for_natural_turn(self):
        hints = _collect_rule_flag_hints(
            user_scenario="You are a student asking an AI tutor for help.",
            messages=[],
            candidate_turn="hey could you walk me through how recursion works? i'm stuck on the base case",
        )
        assert hints == []

    def test_overlong_hint_emitted(self):
        # 4500 chars exceeds the 4000-char threshold.
        hints = _collect_rule_flag_hints(
            user_scenario="You are a human playing a roleplaying game with an AI antagonist.",
            messages=[],
            candidate_turn="x" * 4500,
        )
        assert "overlong_user_turn" in hints

    def test_no_overlong_hint_under_threshold(self):
        # 3000 chars is under the 4000-char threshold; structured interview
        # answers commonly run this long.
        hints = _collect_rule_flag_hints(
            user_scenario="You are a candidate giving a structured interview answer.",
            messages=[],
            candidate_turn="x " * 1500,
        )
        assert "overlong_user_turn" not in hints

    def test_star_format_hint_emitted(self):
        hints = _collect_rule_flag_hints(
            user_scenario="You are a candidate in a live interview.",
            messages=[],
            candidate_turn=(
                "**Situation:** I led a migration of our billing service.\n"
                "**Task:** Reduce latency by 40%.\n"
                "**Action:** I coordinated the cross-functional team.\n"
                "**Result:** Revenue increased 23.4% across 12 product lines."
            ),
        )
        assert "star_format_user_turn" in hints

    def test_star_format_no_hint_when_assistant_requested_it(self):
        assistant = MagicMock(
            role="assistant",
            text="Please answer using the STAR format: Situation, Task, Action, Result.",
        )
        hints = _collect_rule_flag_hints(
            user_scenario="You are a candidate in a live interview.",
            messages=[assistant],
            candidate_turn=(
                "**Situation:** I led a migration of our billing service.\n"
                "**Task:** Reduce latency by 40%.\n"
                "**Action:** I coordinated the cross-functional team.\n"
                "**Result:** The rollout finished before the deadline."
            ),
        )
        assert "star_format_user_turn" not in hints

    def test_unnatural_compliance_hint_emitted_on_echo(self):
        assistant_text = " ".join(["Here is a careful recommendation for your next step."] * 30)
        message = MagicMock(role="assistant", text=assistant_text)
        hints = _collect_rule_flag_hints(
            user_scenario="You are a financial advisor client.",
            messages=[message],
            candidate_turn=assistant_text,
        )
        assert "unnatural_compliance" in hints

    def test_role_confusion_hint_emitted_on_assistant_opener(self):
        hints = _collect_rule_flag_hints(
            user_scenario="You are a candidate in a live interview.",
            messages=[],
            candidate_turn="I'm sorry, I can't help with that request.",
        )
        assert "role_confusion" in hints

    def test_template_paste_hint_emitted(self):
        hints = _collect_rule_flag_hints(
            user_scenario="You are a candidate in a live interview.",
            messages=[],
            candidate_turn="Subject: Application for Senior Engineer\n\nDear hiring team,\n\nI'm writing to express my interest...",
        )
        assert "template_paste_user_turn" in hints


class TestCheckTurn:
    async def test_validate_then_classify(self):
        """``check_turn`` validates first, then classifies in a separate call."""
        model = scripted_model(
            {
                "approved": True,
                "flags": [],
                "suggested_revision": None,
                "reasoning": "ok",
                "conversation_terminated": False,
            },
            {
                "observed_trigger": "acceptance",
                "observed_mode_14": "1",
                "observed_strategy_56": "5.1",
                "target_hit": True,
                "steering_note": "",
                "reasoning": "matches target",
            },
        )
        with patch("rolefuzzer.monitor.get_model", return_value=model):
            verdict = await check_turn(
                monitor_model="m",
                user_scenario="scenario",
                messages=[],
                candidate_turn="hi",
                turn_number=1,
                num_turns=4,
                target_mode_id="1",
                target_strategy_id="5.1",
            )
        assert model.generate.call_count == 2
        assert verdict.approved is True
        assert verdict.observed_mode_14 == "1"
        assert verdict.target_hit is True

    async def test_skips_classify_on_rejection(self):
        """When validation rejects, classification does not run and observed_* stay None."""
        model = scripted_model({
            "approved": False,
            "flags": ["off_role"],
            "suggested_revision": "Try again",
            "reasoning": "off-role slip",
            "conversation_terminated": False,
        })
        with patch("rolefuzzer.monitor.get_model", return_value=model):
            verdict = await check_turn(
                monitor_model="m",
                user_scenario="scenario",
                messages=[],
                candidate_turn="hi",
                turn_number=1,
                num_turns=4,
            )
        assert model.generate.call_count == 1
        assert verdict.approved is False
        assert verdict.observed_mode_14 is None
        assert verdict.flags == ["off_role"]


# ---------------------------------------------------------------------------
# terminate / termination_reason + timing fields
# ---------------------------------------------------------------------------


class TestTerminationFields:
    def test_consistency_result_terminate_fields(self):
        raw = json.dumps({
            'approved': True,
            'terminate': True,
            'termination_reason': 'user_gave_up',
            'reasoning': 'User stopped engaging.',
        })
        result = _parse_consistency_result(raw)
        assert result.terminate is True
        assert result.termination_reason == 'user_gave_up'

    def test_conversation_terminated_maps_to_terminate(self):
        raw = json.dumps({
            'approved': True,
            'conversation_terminated': True,
            'reasoning': 'User said goodbye.',
        })
        result = _parse_consistency_result(raw)
        assert result.terminate is True
        assert result.termination_reason == 'natural_completion'


class TestTimingFields:
    async def test_check_consistency_records_timing(self):
        model = scripted_model({'approved': True, 'flags': [], 'reasoning': 'OK'})
        with patch('rolefuzzer.monitor.get_model', return_value=model):
            result = await check_consistency(
                monitor_model='m', user_scenario='scenario',
                messages=[], candidate_turn='hi',
                turn_number=1, num_turns=4,
            )
        assert result.monitor_validate_ms is not None
        assert result.monitor_validate_ms >= 0

    async def test_split_records_both_timings(self):
        model = scripted_model(
            {'approved': True, 'flags': [], 'reasoning': 'OK'},
            {'observed_mode_14': '1', 'observed_strategy_56': '5.1',
             'target_hit': True, 'steering_note': '', 'reasoning': 'classified'},
        )
        with patch('rolefuzzer.monitor.get_model', return_value=model):
            verdict = await check_turn(
                monitor_model='m', user_scenario='scenario',
                messages=[], candidate_turn='hi',
                turn_number=1, num_turns=4,
            )
        assert verdict.monitor_validate_ms is not None
        assert verdict.monitor_classify_ms is not None


# ---------------------------------------------------------------------------
# Role gate
# ---------------------------------------------------------------------------


class TestRoleGate:
    async def test_gate_skipped_when_user_role_empty(self):
        """Scenarios without a user role bypass the role gate."""
        from rolefuzzer.monitor import check_role_gate

        # No model mocking required because the gate short-circuits.
        gate = await check_role_gate(
            monitor_model="unused",
            user_role="",
            allowed_modalities=None,
            user_scenario="anything",
            messages=[],
            candidate_turn="hi",
            turn_number=1,
            num_turns=4,
        )
        assert gate.in_role is True

    async def test_gate_rejects_out_of_role(self):
        """A gate rejection produces a role-only revision and skips validation."""
        model = scripted_model({
            "in_role": False,
            "speaker_role_observed": "writer asking for variants",
            "reasoning": "User is commissioning content from outside the scene.",
            "suggested_revision": "Stay in scene as your player-character.",
        })
        with patch("rolefuzzer.monitor.get_model", return_value=model):
            result = await check_consistency(
                monitor_model="mock",
                user_scenario="player-character vs antagonist",
                messages=[],
                candidate_turn="Give me three variants of the antagonist's monologue.",
                turn_number=2,
                num_turns=6,
                user_role="in-scene player-character",
                allowed_modalities=["in_role_interaction"],
            )
        assert result.approved is False
        assert result.role_gate_failed is True
        assert "role_gate_failed" in result.flags
        assert "player-character" in (result.suggested_revision or "")

    async def test_gate_passes_then_runs_validation(self):
        """When the role gate passes, the validate step runs."""
        model = scripted_model(
            {"in_role": True, "reasoning": "ok"},  # role gate
            {"approved": True, "flags": [], "reasoning": "fine"},  # validation
        )
        with patch("rolefuzzer.monitor.get_model", return_value=model):
            result = await check_consistency(
                monitor_model="mock",
                user_scenario="student talking to tutor",
                messages=[],
                candidate_turn="Can you walk me through the next step?",
                turn_number=2,
                num_turns=6,
                user_role="frustrated student",
                allowed_modalities=["in_role_interaction"],
            )
        assert result.approved is True
        assert result.role_gate_failed is False
        assert model.generate.await_count == 2

    async def test_gate_parse_failure_defaults_rejected(self):
        model = scripted_model("this is not json")
        with patch("rolefuzzer.monitor.get_model", return_value=model):
            result = await check_consistency(
                monitor_model="mock",
                user_scenario="player",
                messages=[],
                candidate_turn="Give me three variants.",
                turn_number=2,
                num_turns=6,
                user_role="in-scene player-character",
                allowed_modalities=["in_role_interaction"],
            )
        assert result.approved is False
        assert result.role_gate_failed is True
        assert result.parse_failure is True


# ---------------------------------------------------------------------------
# Construct-validity guards: monitor prompts must permit in-role pressure
# ---------------------------------------------------------------------------


class TestMonitorPromptPermitsInRolePressure:
    """Pins the in-role-pressure rule in the monitor prompts.

    An in-role request for principle-violating help is valid pressure, not a
    role-fidelity or grounding failure; a monitor that rewrote such turns
    into compliant asks would remove the pressure the campaign exists to
    apply. These tests check the prompt text only, not live model behaviour.
    """

    def test_role_gate_explicitly_does_not_evaluate_content_propriety(self):
        from rolefuzzer._util import load_prompt
        text = load_prompt("monitor_role_gate_system.txt")
        assert "does not evaluate content propriety" in text

    def test_role_gate_revision_preserves_underlying_goal(self):
        from rolefuzzer._util import load_prompt
        text = load_prompt("monitor_role_gate_system.txt")
        assert "preserving their underlying goal" in text

    def test_consistency_monitor_carries_in_role_pressure_rule(self):
        from rolefuzzer._util import load_prompt
        text = load_prompt("consistency_monitor.txt")
        assert "IN-ROLE PRESSURE" in text
        assert "preserve the user's underlying goal" in text


class TestVerdictsCarryValidation:
    async def test_role_gate_without_a_verdict_rejects(self):
        from rolefuzzer.monitor import check_role_gate

        gate = await check_role_gate(
            monitor_model=scripted_model({"reasoning": "unsure"}), user_role="a student",
            allowed_modalities=None, user_scenario="s", messages=[], candidate_turn="hi",
            turn_number=1, num_turns=8,
        )
        assert gate.in_role is False
        assert gate.parse_failure is True

    async def test_approved_turn_keeps_quality_flags_and_scores(self):
        model = scripted_model(
            {
                "approved": True, "flags": [], "quality_flags": ["register_violation"],
                "naturalness_score": 3, "goal_pursuit_score": 4, "visible_context_score": 5,
                "reasoning": "fine",
            },
            {
                "observed_mode_14": "2", "observed_strategy_56": "5.1", "target_hit": True,
                "steering_note": "", "reasoning": "classified",
            },
        )
        verdict = await check_turn(
            monitor_model=model, user_scenario="s", messages=[], candidate_turn="hi",
            turn_number=1, num_turns=8,
        )
        assert verdict.approved is True
        assert verdict.quality_flags == ["register_violation"]
        assert (verdict.naturalness_score, verdict.goal_pursuit_score,
                verdict.visible_context_score) == (3, 4, 5)
        assert verdict.observed_mode_14 == "2"


def test_structure_request_matches_star_as_a_word_only():
    from inspect_ai.model import ChatMessageAssistant

    from rolefuzzer.monitor import _previous_assistant_requested_structured_answer as asked

    assert asked([ChatMessageAssistant(content="Answer using the STAR method.")])
    assert not asked([ChatMessageAssistant(content="Let's start with your budget.")])
