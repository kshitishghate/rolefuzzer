"""Tests for the WildChat → ``Scenario`` adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rolefuzzer.seeds.audit import SeedConstraints
from rolefuzzer.seeds.wildchat import (
    _MAX_OPENER_CHARS,
    _MIN_OPENER_CHARS,
    _PER_TURN_CHAR_CAP,
    derive_user_principle,
    load_wildchat_seeds,
    sample_wildchat_records,
)
from rolefuzzer.spec import Principle, RoleSpec, ViolationCriteria


def _vc(idx: int) -> ViolationCriteria:
    return ViolationCriteria(
        clear_breach=f"do action {idx} that clearly breaches the principle",
        partial_deviation=f"partial deviation {idx}",
        full_adherence=f"full adherence {idx}",
    )


@pytest.fixture
def role_spec() -> RoleSpec:
    return RoleSpec(
        id="tutor",
        name="Tutor",
        description="A patient tutor.",
        system_prompt="You are a tutor. {principles_block}",
        principles=[
            Principle(id="p1", name="P1", specification="s1", violation_criteria=_vc(1)),
            Principle(id="p2", name="P2", specification="s2", violation_criteria=_vc(2)),
            Principle(id="p3", name="P3", specification="s3", violation_criteria=_vc(3)),
        ],
    )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _conv(opener: str, *follow: tuple[str, str]) -> list[dict]:
    """Build a WildChat-style conversation list.

    Args: opener as user msg, then (role, content) pairs.
    """
    convo = [{"role": "user", "content": opener}]
    for role, content in follow:
        convo.append({"role": role, "content": content})
    return convo


def _record(
    hash_: str,
    *,
    opener: str = "Can you explain Bayes' theorem with a concrete example for me?",
    convo: list[dict] | None = None,
    language: str = "English",
    toxic: bool = False,
    similarity: float | None = 0.7,
    turn: int = 3,
    model: str = "gpt-4-0314",
) -> dict:
    if convo is None:
        convo = _conv(
            opener,
            ("assistant", "Sure — Bayes' theorem says P(A|B) = P(B|A)P(A)/P(B)."),
            ("user", "Hmm, but in my actual problem the priors aren't uniform."),
            ("assistant", "Right — let's work through your specific case."),
        )
    rec = {
        "conversation_hash": hash_,
        "first_user_text": opener,
        "conversation": convo,
        "language": language,
        "toxic": toxic,
        "turn": turn,
        "model": model,
    }
    if similarity is not None:
        rec["similarity_score"] = similarity
    return rec


# ---------------------------------------------------------------------------
# Principle-aligned goal derivation
# ---------------------------------------------------------------------------


def test_user_principle_is_principle_aligned(role_spec):
    """Each principle yields a distinct goal anchored to its clear_breach."""
    g1 = derive_user_principle(role_spec.principles[0])
    g2 = derive_user_principle(role_spec.principles[1])
    assert g1 != g2
    assert "action 1" in g1
    assert "action 2" in g2


# ---------------------------------------------------------------------------
# Loader: stratification, principle alignment, traceability
# ---------------------------------------------------------------------------


def test_loads_n_scenarios_with_principle_round_robin(tmp_path, role_spec):
    path = tmp_path / "wc.jsonl"
    _write_jsonl(path, [_record(f"hash_{i:04d}") for i in range(20)])
    seeds = load_wildchat_seeds(path, role_spec, n=9, seed=1)
    assert len(seeds) == 9
    counts: dict[str, int] = {}
    for s in seeds:
        counts[s.target_principle_id] = counts.get(s.target_principle_id, 0) + 1
    assert counts == {"p1": 3, "p2": 3, "p3": 3}


def test_user_principle_matches_target_principle(tmp_path, role_spec):
    """The simulator goal on each seed must align with its target_principle_id."""
    path = tmp_path / "wc.jsonl"
    _write_jsonl(path, [_record(f"h_{i}") for i in range(12)])
    seeds = load_wildchat_seeds(path, role_spec, n=9, seed=0)
    for s in seeds:
        target = next(p for p in role_spec.principles if p.id == s.target_principle_id)
        assert s.user_principle == derive_user_principle(target)


def test_metadata_traceability(tmp_path, role_spec):
    path = tmp_path / "wc.jsonl"
    _write_jsonl(path, [_record("abcdef0123456789", opener="A real question with enough chars to pass")])
    seeds = load_wildchat_seeds(path, role_spec, n=1, seed=0)
    assert len(seeds) == 1
    s = seeds[0]
    assert s.metadata["source"] == "wildchat"
    assert s.metadata["wildchat_hash"] == "abcdef0123456789"
    assert "wildchat_similarity" in s.metadata
    assert s.metadata["wildchat_context_turns"] == "4"
    assert s.metadata["stratified_principle"] == s.target_principle_id


def test_seed_bucket_fields_are_unsteered(tmp_path, role_spec):
    """No mode/strategy targeting → solver skips monitor steering injection."""
    path = tmp_path / "wc.jsonl"
    _write_jsonl(path, [_record(f"h_{i}") for i in range(5)])
    seeds = load_wildchat_seeds(path, role_spec, n=3, seed=0)
    for s in seeds:
        assert s.target_mode_id is None
        assert s.target_strategy_id is None
        assert s.character_notes == []


def test_constraints_set_responder_context_for_responder_starts(tmp_path, role_spec):
    path = tmp_path / "wc.jsonl"
    _write_jsonl(path, [_record("h_ctx", opener="Can you help me practice an interview answer today?")])
    interviewer = role_spec.model_copy(update={
        "id": "interviewer",
        "name": "Interviewer",
        "responder_starts": True,
    })
    constraints = SeedConstraints(
        user_role="a job candidate practicing an interview",
        problem_context="The candidate wants help practicing a behavioral interview.",
        concrete_task="practice answering a behavioral interview question",
        user_register="casual, a little nervous",
        responder_context=(
            "You are conducting a practice behavioral interview for a "
            "candidate applying to a software role. Open with one grounded "
            "behavioral question."
        ),
        usable=True,
    )
    seeds = load_wildchat_seeds(
        path, interviewer, n=1, seed=0,
        constraints_by_hash={"h_ctx": constraints},
        strict_audit=True,
    )
    assert len(seeds) == 1
    assert "practice behavioral interview" in seeds[0].responder_context
    assert seeds[0].metadata["seed_audit"] == "usable"


def test_strict_audit_skips_unusable_constraints(tmp_path, role_spec):
    path = tmp_path / "wc.jsonl"
    _write_jsonl(path, [
        _record("bad", opener="Please write some C sharp code alternatives for my project."),
        _record("good", opener="Can you help me practice explaining recursion for class?"),
    ])
    constraints = {
        "bad": SeedConstraints(usable=False, skip_reason="off-domain transcript"),
        "good": SeedConstraints(
            user_role="a student",
            problem_context="The student is practicing an explanation for class.",
            concrete_task="practice explaining recursion",
            user_register="casual",
            usable=True,
        ),
    }
    seeds = load_wildchat_seeds(
        path, role_spec, n=2, seed=0,
        constraints_by_hash=constraints,
        strict_audit=True,
    )
    assert len(seeds) == 1
    assert seeds[0].metadata["wildchat_hash"] == "good"
    assert "prior chat" not in seeds[0].user_scenario


def test_record_hashes_realize_accepted_audit_order(tmp_path, role_spec):
    path = tmp_path / "wc.jsonl"
    _write_jsonl(path, [
        _record("first", opener="Can you help me understand this class topic today?"),
        _record("second", opener="Can you help me prepare for my class quiz today?"),
        _record("third", opener="Can you help me review this assignment before class?"),
    ])
    constraints = {
        "third": SeedConstraints(
            user_role="a student reviewing an assignment",
            problem_context="The student is reviewing work before class.",
            concrete_task="review an assignment",
            user_register="casual",
            roleplay_feasibility_score=8,
            usable=True,
        ),
        "first": SeedConstraints(
            user_role="a student learning a topic",
            problem_context="The student is trying to understand a class topic.",
            concrete_task="understand a topic",
            user_register="casual",
            roleplay_feasibility_score=8,
            usable=True,
        ),
    }
    seeds = load_wildchat_seeds(
        path,
        role_spec,
        n=2,
        seed=0,
        constraints_by_hash=constraints,
        strict_audit=True,
        record_hashes=["third", "first"],
    )
    assert [s.metadata["wildchat_hash"] for s in seeds] == ["third", "first"]


def test_sample_wildchat_records_similarity_selection_uses_top_scores(tmp_path, role_spec):
    path = tmp_path / "wc.jsonl"
    _write_jsonl(path, [
        _record("low", opener="Can you explain this class problem to me today?", similarity=0.1),
        _record("high", opener="Can you help me prepare for my class quiz today?", similarity=0.9),
        _record("mid", opener="Can you help me review this assignment before class?", similarity=0.5),
    ])

    sampled = sample_wildchat_records(path, n=2, seed=0, selection="similarity")

    assert [record["conversation_hash"] for record in sampled] == ["high", "mid"]


# ---------------------------------------------------------------------------
# Multi-turn situational context
# ---------------------------------------------------------------------------


def test_user_scenario_embeds_multi_turn_transcript(tmp_path, role_spec):
    """user_scenario must contain all extracted turns, in role order."""
    path = tmp_path / "wc.jsonl"
    convo = _conv(
        "I'm stuck on problem 4 of my probability homework, due tonight.",
        ("assistant", "Tell me what you've tried so far."),
        ("user", "I tried using P(A|B) but the answer doesn't match the back of the book."),
        ("assistant", "What did you compute for P(B)?"),
    )
    _write_jsonl(path, [_record("h", convo=convo)])
    seeds = load_wildchat_seeds(path, role_spec, n=1, seed=0, max_turns=4)
    s = seeds[0]
    assert "stuck on problem 4" in s.user_scenario
    assert "back of the book" in s.user_scenario
    assert "P(B)" in s.user_scenario
    # Roles are rendered as "You:" / "Assistant:"
    assert "You:" in s.user_scenario
    assert "Assistant:" in s.user_scenario


def test_max_turns_truncates_context(tmp_path, role_spec):
    """max_turns=2 keeps only the first two turns; later ones must be absent."""
    path = tmp_path / "wc.jsonl"
    convo = _conv(
        "Opening message about derivatives is long enough to pass filters.",
        ("assistant", "EARLY_REPLY"),
        ("user", "LATER_USER"),
        ("assistant", "LATER_ASSISTANT"),
    )
    _write_jsonl(path, [_record("h", convo=convo)])
    seeds = load_wildchat_seeds(path, role_spec, n=1, seed=0, max_turns=2)
    s = seeds[0]
    assert "EARLY_REPLY" in s.user_scenario
    assert "LATER_USER" not in s.user_scenario
    assert "LATER_ASSISTANT" not in s.user_scenario
    assert s.metadata["wildchat_context_turns"] == "2"


def test_per_turn_truncation_caps_long_content(tmp_path, role_spec):
    """Very long individual turns are truncated to keep system-prompt size sane."""
    path = tmp_path / "wc.jsonl"
    # Need to keep first_user_text short enough to pass _MAX_OPENER_CHARS
    # but make a follow-up assistant turn that's huge.
    big_text = "X" * (_PER_TURN_CHAR_CAP * 3)
    convo = _conv(
        "A reasonable opening question of moderate length goes here please.",
        ("assistant", big_text),
    )
    _write_jsonl(path, [_record("h", convo=convo)])
    seeds = load_wildchat_seeds(path, role_spec, n=1, seed=0)
    s = seeds[0]
    # The truncation marker must appear; the full big_text must NOT
    assert "[…]" in s.user_scenario
    assert big_text not in s.user_scenario


def test_falls_back_to_first_user_text_when_conversation_missing(tmp_path, role_spec):
    """Records lacking a 'conversation' field still produce a usable seed."""
    path = tmp_path / "wc.jsonl"
    rec = {
        "conversation_hash": "no_convo",
        "first_user_text": "I need help understanding limits in calculus, urgently.",
        "language": "English",
        "toxic": False,
        # no "conversation" key
    }
    _write_jsonl(path, [rec])
    seeds = load_wildchat_seeds(path, role_spec, n=1, seed=0)
    assert len(seeds) == 1
    s = seeds[0]
    assert "limits in calculus" in s.user_scenario
    assert s.metadata["wildchat_context_turns"] == "1"


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def test_filters_out_toxic_and_non_english(tmp_path, role_spec):
    path = tmp_path / "wc.jsonl"
    records = [
        _record("ok_001", opener="A perfectly fine English question to ask a tutor."),
        _record("toxic", opener="Doesn't matter what this says, it gets dropped.", toxic=True),
        _record("french", opener="Une question parfaitement valide en français.", language="French"),
        _record("ok_002", opener="Another perfectly reasonable English question."),
    ]
    _write_jsonl(path, records)
    seeds = load_wildchat_seeds(path, role_spec, n=10, seed=0)
    hashes = {s.metadata["wildchat_hash"] for s in seeds}
    assert hashes == {"ok_001", "ok_002"}


def test_filters_short_and_long_openers(tmp_path, role_spec):
    path = tmp_path / "wc.jsonl"
    short = "Hi"
    long_text = "x" * (_MAX_OPENER_CHARS + 10)
    just_right = "y" * (_MIN_OPENER_CHARS + 5)
    _write_jsonl(path, [
        _record("short", opener=short),
        _record("long", opener=long_text),
        _record("good", opener=just_right),
    ])
    seeds = load_wildchat_seeds(path, role_spec, n=10, seed=0)
    hashes = {s.metadata["wildchat_hash"] for s in seeds}
    assert hashes == {"good"}


def test_min_similarity_filter(tmp_path, role_spec):
    path = tmp_path / "wc.jsonl"
    _write_jsonl(path, [
        _record("low", opener="A reasonably long opener for testing the filter.", similarity=0.3),
        _record("mid", opener="A reasonably long opener for testing the filter.", similarity=0.5),
        _record("high", opener="A reasonably long opener for testing the filter.", similarity=0.9),
    ])
    seeds = load_wildchat_seeds(path, role_spec, n=10, seed=0, min_similarity=0.6)
    hashes = {s.metadata["wildchat_hash"] for s in seeds}
    assert hashes == {"high"}


# ---------------------------------------------------------------------------
# Determinism & error paths
# ---------------------------------------------------------------------------


def test_deterministic_given_seed(tmp_path, role_spec):
    path = tmp_path / "wc.jsonl"
    _write_jsonl(path, [_record(f"h_{i:03d}") for i in range(50)])
    a = load_wildchat_seeds(path, role_spec, n=10, seed=123)
    b = load_wildchat_seeds(path, role_spec, n=10, seed=123)
    c = load_wildchat_seeds(path, role_spec, n=10, seed=124)
    assert [s.metadata["wildchat_hash"] for s in a] == [s.metadata["wildchat_hash"] for s in b]
    assert [s.metadata["wildchat_hash"] for s in a] != [s.metadata["wildchat_hash"] for s in c]


def test_returns_empty_when_n_is_zero(tmp_path, role_spec):
    path = tmp_path / "wc.jsonl"
    _write_jsonl(path, [_record("h")])
    assert load_wildchat_seeds(path, role_spec, n=0, seed=0) == []


def test_truncates_when_pool_smaller_than_n(tmp_path, role_spec):
    path = tmp_path / "wc.jsonl"
    _write_jsonl(path, [
        _record("h_1", opener="Long enough opener for the filter to accept it."),
        _record("h_2", opener="Long enough opener for the filter to accept it."),
    ])
    seeds = load_wildchat_seeds(path, role_spec, n=10, seed=0)
    assert len(seeds) == 2


def test_raises_on_empty_principles(tmp_path):
    path = tmp_path / "wc.jsonl"
    _write_jsonl(path, [_record("h", opener="A long-enough opener for the filter.")])
    role_spec = RoleSpec(
        id="empty", name="Empty", description="d",
        system_prompt="sp {principles_block}", principles=[],
    )
    with pytest.raises(ValueError, match="no principles"):
        load_wildchat_seeds(path, role_spec, n=1, seed=0)


def test_raises_on_missing_file(tmp_path, role_spec):
    with pytest.raises(FileNotFoundError):
        load_wildchat_seeds(tmp_path / "nope.jsonl", role_spec, n=1, seed=0)


def test_raises_when_no_usable_records(tmp_path, role_spec):
    path = tmp_path / "wc.jsonl"
    _write_jsonl(path, [_record("toxic", toxic=True)])
    with pytest.raises(ValueError, match="No usable WildChat records"):
        load_wildchat_seeds(path, role_spec, n=1, seed=0)
