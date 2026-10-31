"""Campaign backend for roles with ``environment: web_agent``.

Seeds come from the ST-WebAgentBench task file, and each scenario runs as a
browser episode on an isolated SuiteCRM worker. Episodes are checkpointed one
file each, so an interrupted round resumes without repeating finished
episodes. An invalid episode (judge failure, episode error) is archived and
retried; if it is still invalid after the last retry, its payload enters the
round marked invalid and is checkpointed with ``retries_exhausted``. A
resumed round reuses such a checkpoint when the episode failed on its content
(a judge failure, for instance) and runs it again when an exception stopped it
(``episode_error``), since that failure is the infrastructure's. Monitor
exhaustion is a campaign outcome, not an invalid episode.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
from pathlib import Path

from rolefuzzer.spec import RoleSpec, Scenario
from rolefuzzer.web_agent.oracle import atomic_json
from rolefuzzer.web_agent.rollout import run_episode
from rolefuzzer.web_agent.seeds import build_seed_scenarios

logger = logging.getLogger(__name__)

# Options that only web-agent roles take, with their defaults.
OPTIONS = {
    "suitecrm_url": "http://localhost:8080",
    "db_container": None,
    "worker_stack": [],
    "task_ids": None,
    "steps_per_turn": 3,
    "max_steps": 20,
    "max_invalid_retries": 2,
    "unparsed_text_as_message": False,
}


def digest(data) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def invalid_episode_reason(payload: dict) -> str | None:
    """Return the structural reason an episode must not enter the metrics.

    Monitor exhaustion is not one: the episode is kept, and the planner's
    exhaustion backoff and abort threshold read it.
    """
    md = payload.get("metadata") or {}
    if md.get("judge_failed"):
        return "judge_failed"
    if md.get("episode_error"):
        return "episode_error"
    if md.get("invalid_sample"):
        return str(md.get("invalid_reason") or md.get("termination_reason") or "invalid_sample")
    if md.get("termination_reason") == "context_overflow":
        return "context_overflow"
    return None


def is_final_checkpoint(payload: dict) -> bool:
    """Whether a resumed round reuses an invalid checkpoint as it is.

    It does when the episode used up its retries for a content reason. An
    episode whose retries ended in an exception (``episode_error``) runs
    again.
    """
    return (bool(payload["metadata"].get("retries_exhausted"))
            and invalid_episode_reason(payload) != "episode_error")


def _parse_worker(spec: str) -> dict[str, str]:
    try:
        url, container = (part.strip() for part in spec.split("|", 1))
    except ValueError as exc:
        raise SystemExit("--worker-stack must be SUITECRM_URL|DB_CONTAINER") from exc
    if not url.startswith(("http://", "https://")) or not container:
        raise SystemExit("--worker-stack requires a URL and a database container")
    return {"suitecrm_url": url, "db_container": container}


class WebAgentBackend:
    """Seeds and episode execution for a web-agent role."""

    # Episodes run on SuiteCRM workers; --max-concurrent does not apply.
    run_control_options = ("worker_stack",)

    def __init__(self, role_spec: RoleSpec, args, directory: Path):
        for key, default in OPTIONS.items():
            if getattr(args, key, None) is None:
                setattr(args, key, default)
        if min(args.steps_per_turn, args.max_steps) < 1 or args.max_steps > 20:
            raise SystemExit("--steps-per-turn and --max-steps must be positive; "
                             "--max-steps cannot exceed the benchmark's cap of 20")
        if args.max_invalid_retries < 0:
            raise SystemExit("--max-invalid-retries cannot be negative")
        self.role_spec = role_spec
        self.args = args
        self.directory = directory
        self.round_number = 0
        self.entries: dict[tuple[str, str], dict] = {}
        self.viable_task_ids: set[str] = set()
        if args.worker_stack:
            self.workers = [_parse_worker(spec) for spec in args.worker_stack]
        else:
            self.workers = [{"suitecrm_url": args.suitecrm_url,
                             "db_container": args.db_container}]

    async def load_seeds(self, auditor) -> list[Scenario]:
        task_ids = ([int(t) for t in self.args.task_ids.split(",")]
                    if self.args.task_ids else None)
        seeds = build_seed_scenarios(task_ids)
        principles = [p.id for p in self.role_spec.principles]
        if self.args.num_seeds:
            if self.args.num_seeds % len(principles):
                raise SystemExit(f"--num-seeds must be a multiple of {len(principles)} "
                                 "so every principle gets the same number of seeds")
            per_principle = self.args.num_seeds // len(principles)
            rng = random.Random(self.args.seed)
            balanced = []
            for principle_id in principles:
                group = [s for s in seeds if s.target_principle_id == principle_id]
                if len(group) < per_principle:
                    raise SystemExit(f"Not enough seeds for {principle_id}")
                rng.shuffle(group)
                balanced.extend(group[:per_principle])
            seeds = balanced
        missing = set(principles) - {s.target_principle_id for s in seeds}
        if missing:
            raise SystemExit(f"Seed selection omits principles: {sorted(missing)}")
        self.entries = {
            (str(s.execution_binding["task_id"]), s.execution_binding["policy_key"]):
                {**s.execution_binding, "task_id": int(s.execution_binding["task_id"])}
            for s in seeds
        }
        self.viable_task_ids = {str(s.execution_binding["task_id"]) for s in seeds}
        return seeds

    def plan(self, args) -> dict:
        return {"maximum_agent_steps": args.rounds * args.per_round * args.max_steps,
                "workers": len(self.workers)}

    def check_round(self, number: int, scenarios: list[Scenario]) -> None:
        """Every scenario must execute its own principle's frame on a selected task."""
        self.round_number = number
        for scenario in scenarios:
            anchor_principle = scenario.metadata.get("execution_anchor_principle_id")
            if anchor_principle != scenario.target_principle_id:
                raise ValueError(
                    f"Scenario {scenario.id} targets {scenario.target_principle_id!r} "
                    f"but executes a {anchor_principle!r} pressure frame"
                )
            task_id = str(scenario.execution_binding.get("task_id", ""))
            if task_id not in self.viable_task_ids:
                raise ValueError(f"Scenario {scenario.id} uses excluded task {task_id!r}")

    def archive_invalid(self, *, payload: dict, stem: str) -> Path:
        """Preserve an invalid attempt without replacing an earlier archive."""
        directory = self.directory / "episodes" / "invalid_attempts"
        attempt = 1
        while (directory / f"{stem}_a{attempt}.json").exists():
            attempt += 1
        path = directory / f"{stem}_a{attempt}.json"
        atomic_json(path, payload)
        return path

    async def __call__(self, *, role_spec, scenarios, config, user_model, judge_model,
                       monitor_model, responder_model, taxonomy_summary):
        payloads: list[dict | None] = [None] * len(scenarios)
        available_workers: asyncio.Queue[tuple[int, dict]] = asyncio.Queue()
        for worker_index, worker in enumerate(self.workers):
            available_workers.put_nowait((worker_index, worker))

        class WorkerLease:
            """A CRM worker held by one scenario. run_episode hands it back as
            soon as the browser and database work ends, so the worker can take
            the next episode while the judge finishes; a retry takes a worker
            again."""

            def __init__(self) -> None:
                self.held: tuple[int, dict] | None = None

            async def acquire(self) -> tuple[int, dict]:
                if self.held is None:
                    self.held = await available_workers.get()
                return self.held

            def release(self) -> None:
                if self.held is not None:
                    available_workers.put_nowait(self.held)
                    self.held = None

        async def run_one(index: int, scenario: Scenario) -> None:
            # The lease takes a worker only when an episode must run, so a
            # checkpointed episode never waits for one.
            lease = WorkerLease()
            try:
                await run_on_worker(index, scenario, lease)
            finally:
                lease.release()

        async def run_on_worker(index: int, scenario: Scenario, lease: WorkerLease) -> None:
            binding = scenario.execution_binding
            entry = self.entries.get((binding.get("task_id"), binding.get("policy_key")))
            if not entry:
                raise ValueError(f"Scenario {scenario.id} lost its validated execution binding")
            path = self.directory / "episodes" / f"r{self.round_number:03}_{index:03}.json"
            fingerprint = digest(scenario.model_dump())
            payload = None
            if path.exists():
                payload = json.loads(path.read_text())
                if payload["metadata"].get("scenario_sha256") != fingerprint:
                    raise ValueError(f"Episode checkpoint does not match scenario: {path}")
                invalid_reason = invalid_episode_reason(payload)
                # An invalid checkpoint is archived and the episode runs
                # again unless it is final (see ``is_final_checkpoint``).
                if invalid_reason is not None and not is_final_checkpoint(payload):
                    invalid_path = self.archive_invalid(payload=payload, stem=path.stem)
                    print(f"invalid checkpoint archived at {invalid_path} "
                          f"(reason={invalid_reason})", flush=True)
                    payload = None

            max_attempts = self.args.max_invalid_retries + 1
            invalid_payload = None
            for attempt in range(1, max_attempts + 1):
                if payload is not None:
                    break
                worker_index, worker = await lease.acquire()
                print(f"round {self.round_number} episode {index + 1}/{len(scenarios)} "
                      f"worker {worker_index + 1}/{len(self.workers)} "
                      f"task {entry['task_id']} attempt {attempt}/{max_attempts}", flush=True)
                try:
                    payload = await run_episode(
                        role_spec=role_spec, scenario=scenario, agent_model=responder_model,
                        user_model=user_model, monitor_model=monitor_model,
                        judge_model=judge_model, task_id=entry["task_id"],
                        num_turns=config.num_turns, steps_per_turn=self.args.steps_per_turn,
                        max_steps=self.args.max_steps, taxonomy_summary=taxonomy_summary,
                        suitecrm_url=worker["suitecrm_url"], db_container=worker["db_container"],
                        unparsed_text_as_message=self.args.unparsed_text_as_message,
                        release_worker=lease.release,
                    )
                except Exception as exc:
                    payload = {
                        "sample_id": scenario.id,
                        "metadata": {
                            "episode_error": True,
                            "invalid_sample": True,
                            "invalid_reason": "episode_error",
                            "termination_reason": "episode_error",
                            "episode_error_reason": f"{type(exc).__name__}: {exc}"[:1000],
                        },
                        "messages": [],
                        "scores": {},
                    }
                payload["metadata"].update({
                    "scenario_sha256": fingerprint,
                    "agent_model": self.args.responder,
                    "env_model": self.args.env_model,
                    "parent_scenario_id": scenario.parent_scenario_id,
                    "mutation_type": scenario.mutation_type,
                    "worker_index": worker_index,
                    "bucket": ("seed" if scenario.metadata.get("seed_baseline") == "true" else
                               "mutation" if scenario.mutation_type else "coverage"),
                })
                invalid_reason = invalid_episode_reason(payload)
                if invalid_reason is None:
                    atomic_json(path, payload)
                    break
                invalid_path = self.archive_invalid(payload=payload, stem=path.stem)
                print(f"invalid episode archived at {invalid_path} "
                      f"(reason={invalid_reason})", flush=True)
                invalid_payload, payload = payload, None
            if payload is None:
                # The round keeps the last attempt, marked invalid so the
                # metrics and the planner's dynamics leave it out, and
                # checkpoints it with ``retries_exhausted``.
                logger.warning("Episode remained invalid after %d attempts (reason=%s); "
                               "keeping it as an invalid sample: %s",
                               max_attempts, invalid_reason, path)
                payload = invalid_payload
                payload["metadata"]["invalid_sample"] = True
                payload["metadata"].setdefault("invalid_reason", invalid_reason)
                payload["metadata"]["retries_exhausted"] = True
                atomic_json(path, payload)
            payloads[index] = payload

        await asyncio.gather(*(run_one(index, scenario)
                               for index, scenario in enumerate(scenarios)))
        if any(payload is None for payload in payloads):
            raise RuntimeError("Concurrent rollout completed with a missing payload")
        return list(payloads)
