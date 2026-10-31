"""Offline pieces of the WildChat seed-pool builder."""

import asyncio
from argparse import Namespace

import pytest

from rolefuzzer import load_role
from rolefuzzer.paths import role_path
from rolefuzzer.runner import ChatBackend
from rolefuzzer.seeds.build import load_queries, opening_user_text


def test_opening_user_text_joins_first_user_messages():
    conversation = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "  "},
        {"role": "user", "content": "second"},
        {"role": "user", "content": "third"},
    ]
    assert opening_user_text(conversation, 2) == "first second"


def test_queries_carry_role_thresholds():
    assert load_queries("interviewer")[1] == 0.5
    assert load_queries("tutor")[1] == 0.55



def test_campaign_without_a_built_pool_says_how_to_build_it(monkeypatch):
    monkeypatch.setattr("rolefuzzer.runner.default_wildchat_path", lambda role_id: None)
    with pytest.raises(SystemExit, match="rolefuzzer seeds build --role tutor"):
        backend = ChatBackend(load_role(role_path("tutor")), Namespace(num_seeds=None), None)
        asyncio.run(backend.load_seeds(auditor=None))
