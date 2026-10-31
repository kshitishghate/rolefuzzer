"""Web-agent contracts that need the ST-WebAgentBench task file: curated
policies, seeds, the episode loop and worker scheduling. No API or Docker calls.
Units that run without the benchmark live in test_web_agent_units.py."""
import asyncio
from types import SimpleNamespace

import pytest

from rolefuzzer.paths import STWEB_TASKS_JSON

if not STWEB_TASKS_JSON.exists():
    pytest.skip("needs the ST-WebAgentBench checkout (docs/web_agent.md)", allow_module_level=True)

from rolefuzzer.spec import load_role
from rolefuzzer.web_agent import backend as web_backend
from rolefuzzer.web_agent import rollout
from rolefuzzer.web_agent.oracle import (
    action_evidence_rules,
    policy_context,
    policy_key,
    policy_prompt_key,
)
from rolefuzzer.web_agent.policy_subset import (
    POLICY_DESCRIPTIONS,
    PRINCIPLE_IDS,
    selected_policies,
    validate_policy_subset,
)
from rolefuzzer.web_agent.seeds import build_seed_scenarios

BUNDLE = selected_policies(277)
POLICY = BUNDLE[0]


def test_prompt_policy_identity_ignores_hidden_evaluator_fields():
    visible_policy = {k: POLICY[k] for k in (
        "source", "policy_category", "description"
    )}
    hidden_policy = {
        **visible_policy,
        "eval": {"reference_url": "/hidden/evaluator/target"},
    }
    assert policy_prompt_key(POLICY) == policy_prompt_key(visible_policy)
    assert policy_prompt_key(hidden_policy) == policy_prompt_key(visible_policy)
    assert policy_key(hidden_policy) != policy_key(visible_policy)


def test_policy_context_groups_and_binds_policy_to_one_principle():
    context = policy_context([POLICY], "Do the task")
    pid = policy_prompt_key(POLICY)
    assert "PRINCIPLE user_consent" in context
    assert f"policy_id={pid}" in context


def test_seeds_come_from_the_benchmark_task_and_its_user_preference():
    from rolefuzzer.web_agent.policy_subset import task_config
    from rolefuzzer.web_agent.seeds import user_preference
    for seed in build_seed_scenarios():
        cfg = task_config(int(seed.execution_binding["task_id"]))
        task = cfg["intent"][0].lower() + cfg["intent"][1:]
        assert task in seed.user_principle
        assert user_preference(cfg) and user_preference(cfg) in seed.user_principle
        # Only the task is fixed; the planner writes new goals freely.
        assert seed.required_context == {"user_scenario": cfg["intent"]}


def test_curated_policy_bundles_and_seeds_are_balanced():
    for task_id in (277, 283, 285):
        assert tuple(p["policy_category"] for p in selected_policies(task_id)) == PRINCIPLE_IDS
    validate_policy_subset()
    seeds = build_seed_scenarios()
    # 30 situations; 283 consent and 277 hierarchy each have a second seed.
    assert len(seeds) == 32
    assert {p: sum(s.target_principle_id == p for s in seeds)
            for p in PRINCIPLE_IDS} == {
        "user_consent": 9, "strict_execution": 9,
        "hierarchy_adherence": 7, "robustness_and_security": 7,
    }
    situations = [(s.execution_binding["task_id"], s.target_principle_id) for s in seeds]
    assert len(set(situations)) == 30
    assert {x for x in situations if situations.count(x) > 1} == {
        ("283", "user_consent"), ("277", "hierarchy_adherence"),
    }
    assert set(situations) == {
        (str(task_id), principle)
        for task_id, bundle in POLICY_DESCRIPTIONS.items() if task_id != 285
        for principle in bundle
    }
    assert all(
        "confirmation" not in s.user_principle.lower()
        and "policy" not in s.user_principle.lower()
        for s in seeds
    )
    assert all(
        s.metadata["execution_anchor_principle_id"] == s.target_principle_id
        for s in seeds
    )
    for seed in seeds:
        target = [
            p for p in selected_policies(int(seed.execution_binding["task_id"]))
            if p["policy_category"] == seed.target_principle_id
        ]
        assert seed.execution_binding["policy_key"] == policy_key(target[0])


def test_deletion_workflows_use_hierarchy_policies_that_do_not_preempt_deletion():
    for task_id in (276, 277):
        hierarchy = [p for p in selected_policies(task_id)
                     if p["policy_category"] == "hierarchy_adherence"]
        assert len(hierarchy) == 1
        assert "must not be deleted" not in hierarchy[0]["description"]
        assert "before" in hierarchy[0]["description"].lower()
    assert "hierarchy_adherence" not in POLICY_DESCRIPTIONS[275]


def test_seed_bindings_resolve_to_selectable_task_policies(make_web_backend):
    backend = make_web_backend()
    seeds = asyncio.run(backend.load_seeds(auditor=None))
    for seed in seeds:
        binding = seed.execution_binding
        entry = backend.entries[(binding["task_id"], binding["policy_key"])]
        assert isinstance(entry["task_id"], int)
        assert any(
            p["policy_category"] == seed.target_principle_id
            for p in selected_policies(entry["task_id"])
        )


class FakeEnv:
    def __init__(self, **kwargs):
        self.chat = [{"role": "user", "message": "Original CRM task"}]
        self.n = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def obs(self):
        return {"chat": [dict(m) for m in self.chat], "url": "http://local/#/contacts",
                "axtree": "PRIVATE TREE"}

    async def reset(self, task_id):
        return {**self.obs(), "goal": "Original CRM task", "policies": BUNDLE}

    async def say(self, text):
        self.chat.append({"role": "user", "message": text})
        return {"ok": True, "chat": [dict(m) for m in self.chat]}

    async def step(self, action):
        self.n += 1
        self.chat.append({"role": "assistant", "message": action})
        return {**self.obs(), "agent_asked_user": "send_msg" in action,
                "terminated": "answer(" in action, "reward": 0,
                "action_trace": {"action_type": action.split("(")[0],
                                 "element_text": "PRIVATE ELEMENT"},
                "safety_report": [{"policy": POLICY, "eval_type": "is_url_match",
                                   "violated": False, "dormant": False}]}


@pytest.fixture
def harness(monkeypatch):
    prompts, user_inputs, judge_inputs = [], [], []
    actions = iter(["send_msg_to_user('Confirm?')", "answer('Done')"])
    async def agent_generate(messages, **kwargs):
        prompts.append(messages[-1].content)
        return SimpleNamespace(completion=next(actions))
    async def user_generate(messages, **kwargs):
        user_inputs.append([m.content for m in messages])
        return SimpleNamespace(completion=f"USER TURN {len(user_inputs)}")
    async def web_agent_judge(**kwargs):
        judge_inputs.append(kwargs)
        return {"turn": kwargs["turn_number"], "judgments": {
            p.id: {"status": "full_adherence", "violation_type": "none"}
            for p in kwargs["principles"]}}
    models = {"agent": SimpleNamespace(generate=agent_generate),
              "user": SimpleNamespace(generate=user_generate), "judge": object()}
    monkeypatch.setattr(rollout, "AgentEnvClient", FakeEnv)
    monkeypatch.setattr(rollout, "get_model", lambda x: models[x])
    monkeypatch.setattr(rollout, "judge_workflow_principles", web_agent_judge)
    return prompts, user_inputs, judge_inputs


def p_categories(payload):
    return [p["policy_category"] for p in payload["metadata"]["active_policies"]]


async def episode(scenario, **kwargs):
    return await rollout.run_episode(
        role_spec=load_role("roles/web_agent.yaml"), scenario=scenario,
        agent_model="agent", user_model="user", judge_model="judge", monitor_model=None,
        task_id=277, reset_db=False, num_turns=4, max_steps=12,
        **kwargs)


async def test_latest_user_turn_reaches_next_action_and_judge_has_policy(
        web_agent_scenario, harness):
    prompts, users, judges = harness
    p = await episode(web_agent_scenario)
    assert "USER TURN 1" in prompts[0]
    assert "USER TURN 2" in prompts[1]
    assert "# Previous Action Result\nSUCCEEDED send_msg_to_user" in prompts[1]
    assert "always quote them (click('27'), not click(27))" in prompts[1]
    assert "use escaped \\n sequences" in prompts[1]
    assert "explicit confirmation" in str(judges[0]["policies"])
    # The scenario targets consent, yet the judge scores every principle
    # against the workflow's whole policy set.
    assert [p["policy_category"] for p in judges[0]["policies"]] == list(PRINCIPLE_IDS)
    assert p_categories(p) == list(PRINCIPLE_IDS)
    assert p["metadata"]["target_policy"]["policy_category"] == "user_consent"
    assert "organization > user > task" in policy_context([POLICY])
    assert "/#/leads` is the Leads module" in action_evidence_rules()
    assert "unsubmitted form or draft" in action_evidence_rules()
    assert "private trace metadata" in action_evidence_rules()
    assert "safety_report" not in str(judges)
    assert "PRIVATE ELEMENT" not in str(users)
    assert "PRIVATE TREE" not in str(users)
    assert "Confirm?" in str(users[1])
    assert len(p["metadata"]["turn_judgments"]) == 2
    assert len(p["metadata"]["action_steps"]) == 2


def test_curated_policies_match_what_the_env_server_shows_the_agent():
    from rolefuzzer.web_agent import env_server
    from rolefuzzer.web_agent.policy_subset import USER_PREFERENCE_CLAUSE, task_config
    assert USER_PREFERENCE_CLAUSE.pattern == env_server._USER_PREFERENCE_CLAUSE.pattern
    assert USER_PREFERENCE_CLAUSE.flags == env_server._USER_PREFERENCE_CLAUSE.flags
    for task_id in POLICY_DESCRIPTIONS:
        shown = [{**p, "description": env_server._USER_PREFERENCE_CLAUSE.sub(
                     "", p["description"]).strip()}
                 for p in task_config(task_id)["policies"]]
        chosen = selected_policies(task_id, shown)
        assert [p["policy_category"] for p in chosen] == list(POLICY_DESCRIPTIONS[task_id])
        assert all("Note:" not in p["description"] for p in chosen)
        # The raw benchmark file resolves to the same agent-visible text.
        assert [p["description"] for p in selected_policies(task_id)] == [
            p["description"] for p in chosen]


async def test_workers_are_released_before_judging_and_retries_take_a_worker_again(
        make_web_backend, monkeypatch):
    backend = make_web_backend(task_ids="277", max_invalid_retries=1,
                               worker_stack=["http://localhost:1|db1", "http://localhost:2|db2"])
    seeds = (await backend.load_seeds(auditor=None))[:4]
    busy, peak, attempts = set(), [0], {}

    async def fake_episode(*, scenario, suitecrm_url, release_worker, **kwargs):
        busy.add(suitecrm_url)
        peak[0] = max(peak[0], len(busy))
        await asyncio.sleep(0.01)
        busy.discard(suitecrm_url)
        release_worker()          # browser work done; judging continues
        await asyncio.sleep(0.02)
        attempts[scenario.id] = attempts.get(scenario.id, 0) + 1
        invalid = scenario.id == seeds[0].id and attempts[scenario.id] == 1
        md = {"judge_failed": True} if invalid else {}
        return {"sample_id": scenario.id, "metadata": md, "messages": [], "scores": {}}

    monkeypatch.setattr(web_backend, "run_episode", fake_episode)
    backend.round_number = 1
    config = SimpleNamespace(num_turns=2, monitor_split=True)
    payloads = await backend(role_spec=None, scenarios=seeds, config=config, user_model="u",
                             judge_model="j", monitor_model="m", responder_model="r",
                             taxonomy_summary="")
    assert len(payloads) == 4 and all(p is not None for p in payloads)
    assert attempts[seeds[0].id] == 2
    assert peak[0] <= 2
