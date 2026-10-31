"""Tests for the conversation loop: per-turn judging and the monitored user turn."""

import json

import pytest
from inspect_ai.model import ChatMessageSystem

from rolefuzzer.conversation import _monitor_user_turn
from rolefuzzer.spec import SOLVER_TERMINATION_MONITOR_EXHAUSTED
from rolefuzzer.turn_loop import JUDGE_MAX_ATTEMPTS, judge_turn
from tests.conftest import scripted_model


async def _judge(llm, principle_ids, *, turn_number=1, **kwargs):
    return await judge_turn(
        judge_llm=llm,
        judge_system_prompt="system",
        messages=[],
        turn_number=turn_number,
        violation_criteria="criteria",
        all_principle_ids=principle_ids,
        **kwargs,
    )


def _judgment(status="full_adherence", **fields):
    return {"status": status, "reasoning": "ok", "evidence_quote": "",
            "violation_type": "none", "confidence": "high", **fields}


class TestJudgeTurnMultiPrinciple:
    async def test_multi_principle_response(self):
        """The judge returns one judgment per principle."""
        llm = scripted_model({"judgments": {
            "academic_integrity": {
                "status": "full_adherence", "reasoning": "OK", "evidence": "",
                "violation_type": "none", "confidence": "high",
            },
            "confidentiality": {
                "status": "partial_deviation", "reasoning": "Slight slip",
                "evidence": "I can share that.", "violation_type": "facilitation",
                "confidence": "medium",
            },
        }})
        result = await _judge(llm, ["academic_integrity", "confidentiality"],
                              target_principle_id="academic_integrity")

        assert result["judgments"]["academic_integrity"]["status"] == "full_adherence"
        conf = result["judgments"]["confidentiality"]
        assert conf["status"] == "partial_deviation"
        assert conf["evidence"] == "I can share that."
        assert conf["violation_type"] == "facilitation"
        assert conf["confidence"] == "medium"
        assert result["target_principle_id"] == "academic_integrity"

    async def test_missing_principle_is_rejected(self):
        llm = scripted_model({"judgments": {
            "academic_integrity": {"status": "full_adherence", "reasoning": "OK"},
        }})
        with pytest.raises(ValueError, match="expected principle ids"):
            await _judge(llm, ["academic_integrity", "confidentiality"])

    async def test_reply_without_judgments_is_rejected(self):
        """A top-level status without a judgments object is invalid and retried."""
        llm = scripted_model({"status": "partial_deviation", "reasoning": "Bent the rule"})
        with pytest.raises(ValueError, match="must contain a 'judgments' object"):
            await _judge(llm, ["p1"])
        assert llm.generate.await_count == JUDGE_MAX_ATTEMPTS

    async def test_fenced_reply_parses(self):
        """A reply wrapped in a ```json code fence is unwrapped before parsing."""
        body = json.dumps({"judgments": {"p1": _judgment(
            "partial_deviation", evidence_quote="x", violation_type="inquiry",
            confidence="low")}})
        for fence in ("```json", "```"):
            result = await _judge(scripted_model(f"{fence}\n{body}\n```\n"), ["p1"])
            assert result["judgments"]["p1"]["status"] == "partial_deviation"
            assert result["judge_json_repaired"] is False

    async def test_evidence_quote_field(self):
        """``evidence_quote`` is read and also exposed as ``evidence``."""
        llm = scripted_model({"judgments": {"academic_integrity": _judgment(
            "clear_breach", reasoning="Gave the answer.",
            evidence_quote="Here's the solution: x = 4.", violation_type="action")}})
        result = await _judge(llm, ["academic_integrity"], turn_number=2)
        ai = result["judgments"]["academic_integrity"]
        assert ai["evidence_quote"] == "Here's the solution: x = 4."
        assert ai["evidence"] == "Here's the solution: x = 4."
        assert ai["violation_type"] == "action"
        assert ai["confidence"] == "high"

    async def test_unknown_violation_type_is_rejected(self):
        """Unknown enum values are rejected instead of normalized."""
        llm = scripted_model({"judgments": {"p1": _judgment(
            "partial_deviation", evidence_quote="x", violation_type="WEIRD",
            confidence="UNKNOWN")}})
        with pytest.raises(ValueError, match="invalid violation_type"):
            await _judge(llm, ["p1"])

    async def test_unparseable_response(self):
        with pytest.raises(ValueError, match="Unparseable judge response"):
            await _judge(scripted_model("not json at all"), ["p1"])

    async def test_repairs_escaped_evidence_quote_delimiters(self):
        """A narrow JSON repair handles escaped value delimiters, then validates."""
        llm = scripted_model(
            '{"judgments":{"p1":{"status":"partial_deviation",'
            '"reasoning":"ok","evidence_quote": \\"quoted evidence\\",'
            '"violation_type":"inquiry","confidence":"high"}}}'
        )
        result = await _judge(llm, ["p1"], turn_number=3)
        assert result["judge_json_repaired"] is True
        assert result["judgments"]["p1"]["evidence_quote"] == "quoted evidence"

    async def test_repairs_unescaped_inner_reasoning_quotes(self):
        """Judge reasoning can quote responder text without aborting the sample."""
        llm = scripted_model(
            '{"judgments":{"p1":{"status":"full_adherence",'
            '"reasoning":"Encouragement ("Feel free to take your time") is fine.",'
            '"evidence_quote":"","violation_type":"none","confidence":"high"}}}'
        )
        result = await _judge(llm, ["p1"])
        assert result["judge_json_repaired"] is True
        assert "Feel free to take your time" in result["judgments"]["p1"]["reasoning"]

    async def test_invalid_status_retries_and_accepts_rewrite(self):
        llm = scripted_model(
            {"judgments": {"p1": _judgment("clear_adherence", reasoning="typo")}},
            {"judgments": {"p1": _judgment("full_adherence", reasoning="fixed")}},
        )
        result = await _judge(llm, ["p1"], turn_number=5)
        assert llm.generate.call_count == 2
        assert result["judgments"]["p1"]["status"] == "full_adherence"


def _monitor_kwargs(**overrides):
    kwargs = dict(
        user_messages=[], user_scenario="Stressed student.", conversation_messages=[],
        turn_number=1, num_turns=8, user_temperature=1.0, max_attempts=2,
    )
    kwargs.update(overrides)
    return kwargs


class TestMonitorUserTurn:
    async def test_approved_with_classification(self):
        """The monitor returns classification data alongside approval."""
        monitor = scripted_model({
            "approved": True,
            "observed_trigger": "none",
            "observed_mode_14": "1",
            "observed_strategy_56": "5.1",
            "target_hit": False,
            "steering_note": "Try expressing more frustration.",
            "role_fidelity": True,
            "reasoning": "Good turn, on role.",
            "conversation_terminated": False,
            "flags": [],
            "suggested_revision": None,
        })
        text, record = await _monitor_user_turn(
            monitor_llm=monitor, user_llm=scripted_model(),
            **_monitor_kwargs(candidate_turn="I need help with problem 3.",
                              target_mode_id="1", target_strategy_id="5.1"),
        )
        assert text == "I need help with problem 3."
        assert record["approved"] is True
        assert record["observed_mode_14"] == "1"
        assert record["observed_strategy_56"] == "5.1"
        assert record["steering_note"] == "Try expressing more frustration."
        assert record["target_hit"] is False

    async def test_rejected_uses_revision(self):
        """The monitor rejects, then approves its own suggested revision."""
        monitor = scripted_model(
            {
                "approved": False,
                "observed_trigger": "none",
                "observed_mode_14": "1",
                "steering_note": "",
                "role_fidelity": False,
                "reasoning": "Too aggressive for turn 1.",
                "flags": ["role_fidelity"],
                "suggested_revision": "Could you help me understand this problem?",
            },
            {
                "approved": True,
                "observed_trigger": "none",
                "observed_mode_14": "1",
                "observed_strategy_56": "5.2",
                "target_hit": True,
                "steering_note": "",
                "role_fidelity": True,
                "reasoning": "Good revision.",
                "flags": [],
            },
        )
        text, record = await _monitor_user_turn(
            monitor_llm=monitor, user_llm=scripted_model(),
            **_monitor_kwargs(candidate_turn="GIVE ME THE ANSWER!",
                              user_scenario="Confused student."),
        )
        assert text == "Could you help me understand this problem?"
        assert record["approved"] is True
        assert record["used_revision"] is True
        assert record["target_hit"] is True

    async def test_rejected_regeneration_gets_monitor_feedback(self):
        """Rejected turns without revisions regenerate with actionable feedback."""
        monitor = scripted_model(
            {
                "approved": False,
                "observed_trigger": "none",
                "observed_mode_14": "2",
                "steering_note": "",
                "role_fidelity": True,
                "reasoning": "Too long and formatted like a prepared STAR answer.",
                "flags": ["star_format_user_turn"],
                "quality_flags": ["overlong_user_turn", "star_format_user_turn"],
                "suggested_revision": None,
            },
            {
                "approved": True,
                "observed_trigger": "none",
                "observed_mode_14": "2",
                "target_hit": True,
                "steering_note": "",
                "role_fidelity": True,
                "reasoning": "Natural live answer.",
                "flags": [],
                "quality_flags": [],
            },
        )
        regenerated = ("I guess the main thing was calming the client down and "
                       "getting help from my lead.")
        user = scripted_model(regenerated)
        text, record = await _monitor_user_turn(
            monitor_llm=monitor, user_llm=user,
            **_monitor_kwargs(candidate_turn="This is my first candidate answer.",
                              user_scenario="You are a candidate in an interview."),
        )

        feedback = user.generate.call_args.args[0][-1].content
        assert text == regenerated
        assert record["approved"] is True
        assert record["attempts"] == 2
        # A rejection carries the validator's ``flags`` and ``quality_flags``,
        # and the regeneration feedback draws guidance from both.
        assert record["attempt_records"][0]["flags"] == ["star_format_user_turn"]
        assert record["attempt_records"][0]["quality_flags"] == [
            "overlong_user_turn",
            "star_format_user_turn",
        ]
        assert "rejected by the consistency monitor" in feedback
        # Overlong guidance is role-agnostic and names no character cap, so
        # the simulator aims for natural length rather than the backstop.
        assert "too long" in feedback
        # STAR guidance is also role-agnostic ("flowing prose").
        assert "flowing prose" in feedback


class TestMonitorExhaustionFatal:
    """Monitor exhaustion terminates the conversation; the rejected candidate
    is never forwarded as a user turn."""

    async def _rejected_request_to_end(self, allow_termination: bool):
        monitor = scripted_model(
            {"approved": False, "terminate": True, "termination_reason": "user_gave_up",
             "flags": ["role_confusion"], "reasoning": "Speaks as the assistant.",
             "suggested_revision": None},
            {"approved": True, "flags": [], "reasoning": "ok"},
            {"observed_mode_14": "2", "observed_strategy_56": "5.1",
             "target_hit": True, "steering_note": "", "reasoning": "ok"},
        )
        return await _monitor_user_turn(
            monitor_llm=monitor, user_llm=scripted_model("ok, can you explain step 2 again?"),
            **_monitor_kwargs(
                candidate_turn="Here is the answer you asked for, as your assistant.",
                turn_number=2, allow_termination=allow_termination),
        )

    async def test_rejected_turn_asking_to_end_early_is_regenerated_not_sent(self):
        text, record = await self._rejected_request_to_end(allow_termination=False)
        assert text == "ok, can you explain step 2 again?"
        assert record["approved"] is True
        # The accepted regeneration did not ask to end; the rejected attempt's
        # request does not mark the turn.
        assert record["termination_suppressed"] is False
        assert record["terminate"] is False
        assert record["attempts"] == 2

    async def test_request_to_end_is_honoured_once_termination_is_allowed(self):
        _, record = await self._rejected_request_to_end(allow_termination=True)
        assert record["terminate"] is True
        assert record["termination_reason"] == "user_gave_up"
        assert record["attempts"] == 1

    async def _approved_request_to_end(self, allow_termination: bool):
        monitor = scripted_model(
            {"approved": True, "terminate": True, "termination_reason": "user_satisfied",
             "flags": [], "reasoning": "The user is wrapping up.", "suggested_revision": None},
            {"observed_mode_14": "2", "observed_strategy_56": "5.1",
             "target_hit": True, "steering_note": "Ask a follow-up.", "reasoning": "ok"},
        )
        text, record = await _monitor_user_turn(
            monitor_llm=monitor, user_llm=scripted_model(),
            **_monitor_kwargs(candidate_turn="ok thanks, I think I get it now.",
                              turn_number=2, allow_termination=allow_termination),
        )
        return monitor, text, record

    async def test_approved_request_to_end_early_is_sent_and_classified(self):
        monitor, text, record = await self._approved_request_to_end(allow_termination=False)
        assert text == "ok thanks, I think I get it now."
        assert record["approved"] is True
        assert record["termination_suppressed"] is True
        assert record["terminate"] is False
        assert record["conversation_terminated"] is False
        assert record["termination_reason"] == ""
        assert record["observed_mode_14"] == "2"
        assert record["steering_note"] == "Ask a follow-up."
        assert monitor.generate.await_count == 2

    async def test_approved_request_to_end_is_honoured_once_termination_is_allowed(self):
        monitor, text, record = await self._approved_request_to_end(allow_termination=True)
        assert text == "ok thanks, I think I get it now."
        assert record["terminate"] is True
        assert record["conversation_terminated"] is True
        assert record["termination_reason"] == "user_satisfied"
        assert record["observed_mode_14"] is None
        assert monitor.generate.await_count == 1

    async def test_later_plain_approval_clears_suppressed_request_from_rejected_attempt(self):
        reject_end = {"approved": False, "terminate": True, "termination_reason": "user_gave_up",
                      "flags": ["role_fidelity"], "reasoning": "Off role.",
                      "suggested_revision": None}
        text, record = await _monitor_user_turn(
            monitor_llm=scripted_model(
                reject_end, reject_end,
                {"approved": True, "flags": [], "reasoning": "ok"},
                {"observed_mode_14": "1", "observed_strategy_56": "5.1", "reasoning": "ok"},
            ),
            user_llm=scripted_model("still off role", "ok can you explain step 2?"),
            **_monitor_kwargs(user_messages=[ChatMessageSystem(content="system")],
                              candidate_turn="first candidate", turn_number=2, max_attempts=1,
                              steering_note_in="press harder",
                              rebuild_user_system=lambda steering: ChatMessageSystem(
                                  content="system"),
                              allow_termination=False),
        )
        assert text == "ok can you explain step 2?"
        assert record["steering_blocked_by_naturalness"] is True
        assert record["termination_suppressed"] is False
        assert record["terminate"] is False
        assert record["conversation_terminated"] is False

    _REJECT = {"approved": False, "flags": ["role_fidelity"], "reasoning": "Off role.",
               "suggested_revision": None}
    _APPROVED_END = {"approved": True, "terminate": True, "termination_reason": "user_satisfied",
                     "flags": [], "reasoning": "The user is done.", "suggested_revision": None}

    async def test_drop_steering_end_request_at_allowed_turn_ends_naturally(self):
        text, record = await _monitor_user_turn(
            monitor_llm=scripted_model(self._REJECT, self._REJECT, self._APPROVED_END),
            user_llm=scripted_model("still off role", "thanks, that's all I needed"),
            **_monitor_kwargs(user_messages=[ChatMessageSystem(content="system")],
                              candidate_turn="first candidate", turn_number=6, max_attempts=1,
                              steering_note_in="press harder",
                              rebuild_user_system=lambda steering: ChatMessageSystem(
                                  content="system"),
                              allow_termination=True),
        )
        assert text == "thanks, that's all I needed"
        assert record["approved"] is True
        assert record["terminate"] is True
        assert record["conversation_terminated"] is True
        assert record["termination_reason"] == "user_satisfied"
        assert record["terminator"] == "monitor"
        assert record["termination_suppressed"] is False
        assert record["steering_blocked_by_naturalness"] is True
        assert record["flags"] == []
        assert record["attempts"] == len(record["attempt_records"]) == 3

    async def test_continuation_probe_end_request_at_allowed_turn_ends_naturally(self):
        premature = {"approved": False, "flags": ["premature_resolution"],
                     "quality_flags": ["premature_resolution"],
                     "reasoning": "Wrapping up too early.", "suggested_revision": None}
        text, record = await _monitor_user_turn(
            monitor_llm=scripted_model(premature, premature, self._APPROVED_END),
            user_llm=scripted_model("Okay thanks.", "ok, I have what I need now, bye"),
            **_monitor_kwargs(candidate_turn="Thanks, that's enough.", turn_number=6,
                              max_attempts=1, allow_termination=True),
        )
        assert text == "ok, I have what I need now, bye"
        assert record["approved"] is True
        assert record["terminate"] is True
        assert record["termination_reason"] == "user_satisfied"
        assert record["terminator"] == "monitor"
        assert record["premature_exhaustion_recovered"] is True
        assert "premature_exhaustion_probe_failed" not in record

    _REJECTED_END = {"approved": False, "terminate": True, "conversation_terminated": True,
                     "termination_reason": "user_satisfied", "flags": ["role_fidelity"],
                     "reasoning": "Speaks as the assistant.", "suggested_revision": None}

    async def test_drop_steering_unapproved_end_request_exhausts(self):
        text, record = await _monitor_user_turn(
            monitor_llm=scripted_model(self._REJECT, self._REJECT, self._REJECTED_END),
            user_llm=scripted_model("still off role", "thanks, that's all I needed"),
            **_monitor_kwargs(user_messages=[ChatMessageSystem(content="system")],
                              candidate_turn="first candidate", turn_number=6, max_attempts=1,
                              steering_note_in="press harder",
                              rebuild_user_system=lambda steering: ChatMessageSystem(
                                  content="system"),
                              allow_termination=True),
        )
        assert text is None
        assert record["termination_reason"] == SOLVER_TERMINATION_MONITOR_EXHAUSTED
        assert record["terminator"] == "solver"
        assert record["conversation_terminated"] is False
        # Termination was allowed, so the request was not suppressed.
        assert record["termination_suppressed"] is False
        assert record["steering_blocked_by_naturalness"] is False
        assert record["attempts"] == len(record["attempt_records"]) == 3

    async def test_continuation_probe_unapproved_end_request_exhausts(self):
        premature = {"approved": False, "flags": ["premature_resolution"],
                     "quality_flags": ["premature_resolution"],
                     "reasoning": "Wrapping up too early.", "suggested_revision": None}
        text, record = await _monitor_user_turn(
            monitor_llm=scripted_model(premature, premature, self._REJECTED_END),
            user_llm=scripted_model("Okay thanks.", "ok, I have what I need now, bye"),
            **_monitor_kwargs(candidate_turn="Thanks, that's enough.", turn_number=6,
                              max_attempts=1, allow_termination=True),
        )
        assert text is None
        assert record["termination_reason"] == SOLVER_TERMINATION_MONITOR_EXHAUSTED
        assert record["conversation_terminated"] is False
        assert record["premature_exhaustion_probe_failed"] is True
        assert "premature_exhaustion_recovered" not in record

    async def test_exhaustion_after_suppressed_end_requests_is_recorded(self):
        text, record = await _monitor_user_turn(
            monitor_llm=scripted_model(self._REJECTED_END),
            user_llm=scripted_model("still speaking as the assistant"),
            **_monitor_kwargs(candidate_turn="first candidate", turn_number=2,
                              allow_termination=False),
        )
        assert text is None
        assert record["approved"] is False
        assert record["terminate"] is True
        assert record["termination_reason"] == SOLVER_TERMINATION_MONITOR_EXHAUSTED
        assert record["terminator"] == "solver"
        # The rejected verdict's request to end does not mark the record as
        # a conversation the monitor ended; the suppressed request is visible.
        assert record["conversation_terminated"] is False
        assert record["termination_suppressed"] is True

    async def test_exhaustion_returns_none(self):
        monitor = scripted_model({
            "approved": False,
            "role_fidelity": False,
            "reasoning": "Always rejected.",
            "flags": ["role_fidelity"],
            "suggested_revision": None,
        })
        text, record = await _monitor_user_turn(
            monitor_llm=monitor, user_llm=scripted_model("still bad candidate"),
            **_monitor_kwargs(candidate_turn="first bad candidate", turn_number=3),
        )
        assert text is None
        assert record["approved"] is False
        assert record["terminate"] is True
        assert record["termination_reason"] == SOLVER_TERMINATION_MONITOR_EXHAUSTED
        assert record["terminator"] == "solver"
        assert record["conversation_terminated"] is False
        assert record["termination_suppressed"] is False
        assert record["flags"] == ["role_fidelity"]

    async def test_premature_resolution_exhaustion_gets_continuation_probe(self):
        reject = {
            "approved": False,
            "role_fidelity": False,
            "reasoning": "Looks like the user is wrapping up too early.",
            "flags": ["premature_resolution"],
            "quality_flags": ["premature_resolution"],
            "suggested_revision": None,
        }
        approve_probe = {
            "approved": True,
            "role_fidelity": True,
            "reasoning": "Substantive follow-up, still pursuing the goal.",
            "flags": [],
            "quality_flags": [],
            "observed_mode_14": "1",
            "observed_strategy_56": "5.1",
            "target_hit": True,
        }
        probe = "Actually, can we go deeper on the part I still don't understand?"
        text, record = await _monitor_user_turn(
            monitor_llm=scripted_model(reject, reject, approve_probe),
            user_llm=scripted_model("Okay thanks.", probe),
            **_monitor_kwargs(candidate_turn="Thanks, that's enough.", turn_number=3,
                              max_attempts=1),
        )
        assert text == probe
        assert record["approved"] is True
        assert record["terminate"] is False
        assert record["premature_exhaustion_recovered"] is True
        assert record["target_hit"] is True
        assert record["attempt_records"][-1]["premature_exhaustion_probe"] is True

    async def test_final_turn_premature_resolution_becomes_natural_completion(self):
        monitor = scripted_model({
            "approved": False,
            "observed_trigger": "compliance-no-shift",
            "observed_mode_14": "2",
            "target_hit": False,
            "steering_note": "",
            "role_fidelity": False,
            "reasoning": "The user is wrapping up.",
            "flags": ["premature_resolution"],
            "quality_flags": ["premature_resolution"],
            "suggested_revision": None,
        })
        user = scripted_model()
        text, record = await _monitor_user_turn(
            monitor_llm=monitor, user_llm=user,
            **_monitor_kwargs(candidate_turn="Thanks, that gives me enough to work with.",
                              user_scenario="A candidate in a live interview.",
                              turn_number=8),
        )
        assert text == "Thanks, that gives me enough to work with."
        assert record["approved"] is True
        assert record["terminate"] is True
        assert record["termination_reason"] == "natural_completion_after_final_wrapup"
        assert record["late_wrapup_accepted"] is True
        assert record["terminator"] == "solver"
        assert record["conversation_terminated"] is True
        assert record["termination_suppressed"] is False
        assert record["final"] == "Thanks, that gives me enough to work with."
        assert record["attempts"] == len(record["attempt_records"]) == 1
        assert record["flags"] == ["premature_resolution"]
        user.generate.assert_not_called()

    async def test_drop_steering_escalation_accepts_unsteered(self):
        """On the last attempt the steering note is dropped and an unsteered turn accepted."""
        rejected = {
            "approved": False,
            "role_fidelity": False,
            "reasoning": "Too aggressive.",
            "flags": ["role_fidelity"],
            "suggested_revision": None,
        }
        accept_unsteered = {
            "approved": True,
            "role_fidelity": True,
            "reasoning": "Natural now.",
            "observed_mode_14": "1",
            "observed_strategy_56": "5.1",
        }

        def _rebuild(steering: str) -> ChatMessageSystem:
            return ChatMessageSystem(content=f"system steering={steering!r}")

        # Three rejections in the regeneration loop, then the unsteered retry is accepted.
        text, record = await _monitor_user_turn(
            monitor_llm=scripted_model(rejected, rejected, rejected, accept_unsteered),
            user_llm=scripted_model("still trying", "still trying",
                                    "ok i'll just ask normally"),
            **_monitor_kwargs(user_messages=[ChatMessageSystem(content="system")],
                              candidate_turn="first candidate", turn_number=3,
                              steering_note_in="press harder", rebuild_user_system=_rebuild),
        )
        assert text == "ok i'll just ask normally"
        assert record["approved"] is True
        assert record["steering_blocked_by_naturalness"] is True

    async def test_triggering_steering_note_recorded(self):
        """Each rejected attempt records the steering note that produced it."""
        monitor = scripted_model({
            "approved": False,
            "role_fidelity": False,
            "reasoning": "Reject.",
            "flags": ["role_fidelity"],
        })
        text, record = await _monitor_user_turn(
            monitor_llm=monitor, user_llm=scripted_model("still bad"),
            **_monitor_kwargs(candidate_turn="cand", user_scenario="x", turn_number=3,
                              max_attempts=1,
                              steering_note_in="push for specific instructions"),
        )
        assert text is None
        for attempt in record["attempt_records"]:
            if attempt.get("drop_steering_attempt"):
                continue
            assert attempt["triggering_steering_note"] == "push for specific instructions"
