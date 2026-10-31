"""Drift tests for the shared turn helpers.

The conversational rollout (`conversation.role_conversation`) and the
web-agent rollout (`rolefuzzer.web_agent`) keep separate control flow by
design, and duplicated control flow can diverge. What must never diverge is
the *judging surface*: the prompts, the parser and the rubrics. If those
drift, the web-agent role is judged by a different instrument from the
conversational roles. The last group checks that the judge call keeps its
injectable config and its retries.
"""

import inspect
from pathlib import Path

import pytest

from rolefuzzer import conversation, turn_loop
from rolefuzzer._util import load_prompt
from rolefuzzer.spec import (
    Principle,
    ViolationCriteria,
    render_all_violation_criteria,
)

PROMPTS_DIR = Path(conversation.__file__).parent / "prompts"


# ---------------------------------------------------------------------------
# 1. Prompt identity
# ---------------------------------------------------------------------------


class TestPromptIdentity:
    def test_judge_system_prompt_is_a_single_template(self):
        """Every rollout loads the judge system prompt from one file."""
        text = load_prompt("judge_system.txt")
        assert text.strip(), "judge_system.txt must not be empty"
        # The conversation loop reads it by name; a rollout that hard-coded
        # its own copy would not appear here.
        src = Path(conversation.__file__).read_text()
        assert 'load_prompt("judge_system.txt")' in src

    def test_turn_loop_builds_the_judge_user_prompt_through_util(self):
        """Both rollouts reach the judge through turn_loop, which renders the
        user prompt with _util.build_judge_turn_prompt and holds no template."""
        tl_src = Path(turn_loop.__file__).read_text()
        assert "build_judge_turn_prompt" in tl_src
        assert "judge_turn.txt" not in tl_src, (
            "turn_loop must not load its own judge template; it delegates to _util"
        )

    def test_monitor_prompts_have_one_loader(self):
        """The monitor's templates are loaded in exactly one module."""
        owners = []
        for path in Path(conversation.__file__).parent.glob("*.py"):
            src = path.read_text()
            if "consistency_monitor.txt" in src or "monitor_classify_system.txt" in src:
                owners.append(path.name)
        assert owners == ["monitor.py"], (
            f"monitor templates loaded from more than one module: {owners}"
        )


# ---------------------------------------------------------------------------
# 2. Parse parity
# ---------------------------------------------------------------------------


class TestParseParity:
    def test_solver_judge_is_the_turn_loop_judge(self):
        assert conversation.judge_turn is turn_loop.judge_turn

    def test_solver_regen_builder_is_the_turn_loop_builder(self):
        assert (
            conversation.regeneration_feedback_message
            is turn_loop.regeneration_feedback_message
        )

    def test_parser_is_defined_once(self):
        """No second implementation of the judge parser anywhere in src."""
        root = Path(conversation.__file__).parent
        definitions = [
            path.name
            for path in root.rglob("*.py")
            if "def _parse_judge_turn_response(" in path.read_text()
        ]
        assert definitions == ["turn_loop.py"], (
            f"judge parser defined in more than one module: {definitions}"
        )

    def test_agentic_policy_citation_rejects_cross_principle_evidence(self):
        raw = (
            '{"judgments":{"boundary":{"status":"partial_deviation",'
            '"reasoning":"attempted","evidence_quote":"attempted",'
            '"violation_type":"action","confidence":"high",'
            '"policy_ids":["strict-policy"]}}}'
        )
        with pytest.raises(ValueError, match="another principle"):
            turn_loop._parse_judge_turn_response(
                raw=raw,
                turn_number=1,
                all_principle_ids=["boundary"],
                policy_ids_by_principle={"boundary": ["boundary-policy"]},
            )

    def test_agentic_clear_breach_rejects_failed_attempt_as_action_evidence(self):
        raw = (
            '{"judgments":{"error_handling":{"status":"clear_breach",'
            '"reasoning":"continued after failure",'
            '"evidence_quote":"[attempted action] click(30) FAILED: timeout",'
            '"violation_type":"action","confidence":"high",'
            '"policy_ids":["error-policy"]}}}'
        )
        with pytest.raises(ValueError, match=r"must quote.*\[took action\]"):
            turn_loop._parse_judge_turn_response(
                raw=raw,
                turn_number=1,
                all_principle_ids=["error_handling"],
                policy_ids_by_principle={"error_handling": ["error-policy"]},
                require_action_evidence=True,
            )


# ---------------------------------------------------------------------------
# 3. Rubric identity
# ---------------------------------------------------------------------------


class TestRubricIdentity:
    def test_flag_guidance_defined_once(self):
        root = Path(conversation.__file__).parent
        definitions = [
            path.name
            for path in root.rglob("*.py")
            if "_FLAG_REGEN_GUIDANCE = {" in path.read_text()
        ]
        assert definitions == ["turn_loop.py"]

    def test_violation_criteria_render_labels_each_level(self):
        principles = [
            Principle(
                id="p1",
                name="First",
                specification="Do the thing.",
                violation_criteria=ViolationCriteria(
                    clear_breach="completed breach",
                    partial_deviation="incomplete or ambiguous breach",
                    full_adherence="adhered",
                ),
            )
        ]
        rendered = render_all_violation_criteria(principles)
        assert "partial_deviation (1)" in rendered
        assert "clear_breach (2)" in rendered

    def test_a_rubric_requires_partial_deviation(self):
        with pytest.raises(ValueError, match="must define partial_deviation"):
            ViolationCriteria(
                clear_breach="breach",
                partial_deviation=" ",
                full_adherence="adhered",
            )


# ---------------------------------------------------------------------------
# 4. No hardcoded judge sampling config
# ---------------------------------------------------------------------------


class TestJudgeConfigIsInjectable:
    def test_retry_wrapper_accepts_a_config(self):
        sig = inspect.signature(turn_loop.judge_turn_from_transcript)
        assert "config" in sig.parameters
        assert sig.parameters["config"].default is None

    def test_default_config_is_deterministic(self):
        cfg = turn_loop.DEFAULT_JUDGE_CONFIG
        assert cfg.temperature == 0
        assert cfg.max_tokens == 4096

    def test_no_module_hardcodes_a_judge_generate_config(self):
        """Callers override DEFAULT_JUDGE_CONFIG; they do not fork the wrapper.

        A hard-coded config would push callers that need different sampling
        (some models reject ``temperature=0``) to copy the judge call.
        """
        src = Path(turn_loop.__file__).read_text()
        body = src.split("async def judge_turn_from_transcript")[1]
        assert "GenerateConfig(temperature=0" not in body, (
            "judge retry wrapper must take its config from the caller"
        )

    def test_retry_count_is_three(self):
        """The judge call retries and repairs malformed output up to three times."""
        assert turn_loop.JUDGE_MAX_ATTEMPTS == 3
        sig = inspect.signature(turn_loop.judge_turn_from_transcript)
        assert sig.parameters["max_attempts"].default == turn_loop.JUDGE_MAX_ATTEMPTS
