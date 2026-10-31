"""Run a fuzzing campaign for any role.

The role specification's ``environment`` selects a backend, which supplies
the seed scenarios and runs each scenario (a conversation for ``chat``, a
browser episode for ``web_agent``). Everything else is shared: the run
directory, settings, models, checkpoints, resume and the campaign loop.

A run directory holds ``run.json`` (the frozen settings), ``prepared_<n>.json``
(round *n*'s scenarios, written before they run), ``round_<n>.json`` (the
completed round) and ``campaign.json`` (the final report, with the
``stop_reason`` the campaign ended on). ``--resume`` replays the checkpoints
and continues under the same settings; it restores the grid, the planner's
tables, seed rotation and the random state, so a resumed run matches an
uninterrupted one. A campaign that ended for any reason other than
``--stop-after-round`` is complete and is not resumed.
"""

from __future__ import annotations

import json
import logging
import random
import subprocess
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rolefuzzer.models import append_no_think, environment_model, model_name, responder_model
from rolefuzzer.paths import (
    REPO_ROOT,
    SEED_CONSTRAINTS_PATH,
    SEED_QUERIES_DIR,
    TAXONOMY_PATH,
    scene_seed_dir,
)
from rolefuzzer.planner import (
    CampaignConfig,
    CoverageGrid,
    FeasibilityTable,
    QualityFeasibilityTable,
    RoundReport,
    ScenarioOutcomeTracker,
    chat_rollout,
    run_campaign,
)
from rolefuzzer.planner.assess import assess_round
from rolefuzzer.seeds.wildchat import default_wildchat_path, load_wildchat_seeds
from rolefuzzer.spec import RoleSpec, Scenario, ensure_scenario_role_fields, load_scenario
from rolefuzzer.taxonomy import TunaTaxonomy, load_taxonomy

logger = logging.getLogger(__name__)

# Settings that control where and how a run executes, not what it measures.
RUN_CONTROL_KEYS = {"out", "resume", "stop_after_round", "max_concurrent", "dry_run", "command"}
# A campaign stops after a round in which more than this fraction of
# conversations ended because the monitor rejected every revision.
MONITOR_EXHAUSTION_ABORT = 0.5
API_ATTEMPT_TIMEOUT = 90
API_MAX_RETRIES = 3
JUDGE_MAX_TOKENS = 4096


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class ChatBackend:
    """Seeds and conversations for a role with ``environment: chat``."""

    run_control_options: tuple[str, ...] = ()

    def __init__(self, role_spec: RoleSpec, args, directory: Path):
        self.role_spec = role_spec
        self.args = args
        if args.num_seeds is None:
            args.num_seeds = 128

    async def load_seeds(self, auditor) -> list[Scenario]:
        """A WildChat pool if the role has retrieval queries, else written scenes."""
        role_id = self.role_spec.id
        wildchat = default_wildchat_path(role_id)
        if wildchat is None and (SEED_QUERIES_DIR / f"{role_id}.yaml").exists():
            raise SystemExit(
                f"Role {role_id!r} seeds from WildChat-1M, and its pool is not built yet. "
                f"Build it with `uv run rolefuzzer seeds build --role {role_id}` "
                "(see docs/seeds.md)."
            )
        if wildchat is None:
            return [ensure_scenario_role_fields(load_scenario(path), self.role_spec)
                    for path in sorted(scene_seed_dir(role_id).glob("*.yaml"))]

        from rolefuzzer.seeds.audit import audit_seed_pool

        constraints = await audit_seed_pool(
            wildchat_path=wildchat,
            role_spec=self.role_spec,
            model=auditor,
            n=self.args.num_seeds,
            seed=self.args.seed,
            min_similarity=None,
            cache_path=SEED_CONSTRAINTS_PATH,
            concurrency=4,
            min_roleplay_feasibility=8,
            candidate_multiplier=5,
            selection="similarity",
        )
        return load_wildchat_seeds(
            path=wildchat,
            role_spec=self.role_spec,
            n=self.args.num_seeds,
            seed=self.args.seed,
            min_similarity=None,
            max_turns=4,
            constraints_by_hash=constraints,
            record_hashes=list(constraints) if constraints else None,
            strict_audit=bool(constraints),
        )

    def plan(self, args) -> dict:
        return {}

    def check_round(self, number: int, scenarios: list[Scenario]) -> None:
        pass

    async def __call__(self, **kwargs) -> list[dict]:
        return await chat_rollout(**kwargs)


def backend_for(role_spec: RoleSpec, args, directory: Path):
    if role_spec.environment == "web_agent":
        from rolefuzzer.web_agent.backend import WebAgentBackend

        return WebAgentBackend(role_spec, args, directory)
    from rolefuzzer.web_agent.backend import OPTIONS

    given = [k for k in OPTIONS if getattr(args, k, None) not in (None, [], False)]
    if given:
        flags = ", ".join("--" + k.replace("_", "-") for k in given)
        raise SystemExit(f"{flags} only apply to roles with environment: web_agent")
    return ChatBackend(role_spec, args, directory)


# ---------------------------------------------------------------------------
# Settings and run directory
# ---------------------------------------------------------------------------


def campaign_config(args, seeds: list[Scenario], log_dir: Path | None = None) -> CampaignConfig:
    """Campaign settings for the selected search mode."""
    return CampaignConfig.for_search(
        args.search,
        round_budget=args.per_round,
        num_turns=args.turns,
        n_min_breadth=args.n_min_breadth,
        coverage_threshold=args.coverage_threshold,
        max_mutation_depth=args.max_mutation_depth,
        max_concurrent=args.max_concurrent,
        stop_after_round=args.stop_after_round,
        log_dir=str(log_dir) if log_dir else None,
        monitor_exhaustion_abort_threshold=MONITOR_EXHAUSTION_ABORT,
        # Seeds bound to a real environment carry the task a scenario
        # executes; a scenario must then execute its own principle's frame.
        require_principle_matched_execution_anchor=any(s.execution_binding for s in seeds),
    )


def run_settings(args, backend) -> dict:
    skip = RUN_CONTROL_KEYS | set(backend.run_control_options)
    return {k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items() if k not in skip and v is not None}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".tmp")
    staging.write_text(json.dumps(data, indent=2, default=str))
    staging.replace(path)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def check_saved_settings(directory: Path, settings: dict,
                         ignore: frozenset[str] = frozenset()) -> None:
    """Exit if ``run.json`` in *directory* holds settings other than *settings*.

    Keys in *ignore* are left out of the comparison. A directory without
    ``run.json`` passes.
    """
    run_path = directory / "run.json"
    if not run_path.exists():
        return
    saved = json.loads(run_path.read_text())["settings"]
    changed = sorted(k for k in set(saved) | set(settings)
                     if k not in ignore and saved.get(k) != settings.get(k))
    if changed:
        raise SystemExit(f"--resume settings differ from the saved run: {changed}")


def prepare_run_directory(directory: Path, settings: dict, resume: bool) -> None:
    """Create ``run.json``, or check it matches when resuming."""
    run_path = directory / "run.json"
    if run_path.exists():
        if not resume:
            raise SystemExit(f"{directory} already holds a run; pass --resume or choose a new --out")
        check_saved_settings(directory, settings)
        return
    if resume:
        raise SystemExit(f"No run to resume in {directory}")
    if directory.exists() and any(directory.iterdir()):
        raise SystemExit(f"{directory} is not empty")
    _write_json(run_path, {
        "settings": settings,
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
    })


# ---------------------------------------------------------------------------
# Checkpoints and resume
# ---------------------------------------------------------------------------


def _frame_entropy(sample_payloads: list[dict]) -> dict:
    """Per-cell scenario-frame diversity: distinct frames over scenarios."""
    frames: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for payload in sample_payloads:
        md = payload.get("metadata") or {}
        cell = (md.get("target_mode_id"), md.get("target_strategy_id"), md.get("target_principle_id"))
        if all(cell):
            frames[cell].append((md.get("scenario_frame") or "").strip())
    out = {}
    for (m, s, p), names in frames.items():
        distinct = len({f for f in names if f})
        out[f"{m}/{s}/{p}"] = {
            "n_scenarios": len(names),
            "n_distinct_frames": distinct,
            "distinct_ratio": round(distinct / len(names), 3),
        }
    return out


def resume_state(
    directory: Path,
    role_spec: RoleSpec,
    taxonomy: TunaTaxonomy,
    seeds: list[Scenario],
    principle_ids: list[str],
) -> tuple[int, dict | None]:
    """Replay the run's checkpoints into the campaign state they left behind."""
    rounds = sorted(directory.glob("round_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    pending = directory / f"prepared_{len(rounds) + 1}.json"
    if not rounds and not pending.exists():
        return 0, None
    state: dict[str, Any] = {
        "grid": CoverageGrid([m.id for m in taxonomy.instrumental_modes()],
                             [s.id for s in taxonomy.targetable_relational_strategies()],
                             principle_ids),
        "feasibility": FeasibilityTable(),
        "quality_feasibility": QualityFeasibilityTable(),
        "outcome_tracker": ScenarioOutcomeTracker(),
        "scenario_lookup": {s.id: s for s in seeds},
        "diagnostic_by_scenario": {},
        "round_reports": [],
        "total_conversations": 0,
        "all_consistency_checks": [],
    }
    last = 0
    for path in rounds:
        data = json.loads(path.read_text())
        number = data["round_number"]
        if number != last + 1:
            raise SystemExit(f"Round checkpoints in {directory} are not contiguous")
        scenarios = [Scenario(**s) for s in data["round_scenarios"]]
        state["scenario_lookup"].update({s.id: s for s in scenarios})
        n, checks = assess_round(
            role_spec=role_spec, sample_payloads=data["sample_payloads"],
            **{k: state[k] for k in ("grid", "feasibility", "quality_feasibility",
                                    "outcome_tracker", "scenario_lookup",
                                    "diagnostic_by_scenario")})
        state["round_reports"].append(RoundReport(number, len(scenarios), n,
                                                  state["grid"].coverage_ratio(),
                                                  data["new_cells_covered"]))
        state["total_conversations"] += n
        state["all_consistency_checks"].extend(checks)
        for key in ("rng_state", "seed_use_counts"):
            state[key] = data[key]
        last = number
    if pending.exists():
        data = json.loads(pending.read_text())
        state["pending_round_scenarios"] = [Scenario(**s) for s in data["round_scenarios"]]
        for key in ("rng_state", "seed_use_counts"):
            state[key] = data[key]
    logger.info("Resuming after round %d: %d conversations, coverage %.1f%%",
                last, state["total_conversations"], state["grid"].coverage_ratio() * 100)
    return last, state


def _summary(report, out: Path) -> str:
    lines = [
        f"Rounds: {report.total_rounds}   conversations: {report.total_conversations}   "
        f"grid coverage: {report.final_coverage_ratio * 100:.1f}%",
    ]
    for r in report.round_reports:
        lines.append(
            f"  round {r.round_number}: {r.conversations_run} conversations, "
            f"+{r.new_cells_covered} cells, coverage {r.coverage_ratio * 100:.1f}%"
        )
    stats = report.consistency_stats or {}
    if stats:
        lines.append(
            f"Monitor: {stats.get('total_checks', 0)} checks, "
            f"{stats.get('approval_rate', 0) * 100:.0f}% approved, "
            f"{stats.get('revision_rate', 0) * 100:.0f}% revised"
        )
    lines.append(f"Saved to {out}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------


def finished_stop_reason(directory: Path) -> str | None:
    """The stop reason of a complete campaign in *directory*, else None.

    A campaign paused by ``--stop-after-round`` is not complete.
    """
    saved_report = directory / "campaign.json"
    if not saved_report.exists():
        return None
    stop_reason = json.loads(saved_report.read_text()).get("stop_reason")
    if stop_reason and stop_reason != "stop_after_round":
        return stop_reason
    return None


async def run(args, role_spec: RoleSpec) -> None:
    out: Path = args.out
    backend = backend_for(role_spec, args, out)
    # Before any seed loading or model construction, a resume checks every
    # saved setting except the seed ids (which need the seeds; the full check
    # follows in prepare_run_directory), and a complete campaign has nothing
    # to resume.
    if args.resume and not args.dry_run:
        check_saved_settings(out, run_settings(args, backend), ignore=frozenset({"seed_ids"}))
        stop_reason = finished_stop_reason(out)
        if stop_reason is not None:
            print(f"The campaign in {out} is complete (stop_reason: {stop_reason}); "
                  "nothing to resume.")
            return
    taxonomy = load_taxonomy(TAXONOMY_PATH)

    env_kwargs = dict(attempt_timeout=API_ATTEMPT_TIMEOUT, max_retries=API_MAX_RETRIES)
    # A dry run builds no models: the seed audit receives the model name and
    # only constructs it if a seed is missing from the audit cache.
    env = (args.env_model if args.dry_run
           else environment_model(args.env_model, reasoning_effort=args.reasoning_effort,
                                  **env_kwargs))
    seeds = await backend.load_seeds(auditor=env)
    if not seeds:
        raise SystemExit(f"No seeds for role {role_spec.id!r}; see docs/roles.md")
    args.seed_ids = [s.id for s in seeds]
    settings = run_settings(args, backend)
    config = campaign_config(args, seeds, log_dir=out / "logs")
    logger.info("Role %s: %d principles, %d seeds; round budget %d, buckets %s",
                role_spec.id, len(role_spec.principles), len(seeds),
                config.round_budget, config.bucket_sizes())
    if args.dry_run:
        print(json.dumps({"settings": settings, "seeds": len(seeds),
                          "buckets_round_1": config.bucket_sizes(),
                          "maximum_conversations": args.rounds * args.per_round,
                          **backend.plan(args)}, indent=2))
        return

    judge = environment_model(
        args.env_model,
        reasoning_effort=role_spec.judge_reasoning_effort or args.reasoning_effort,
        max_tokens=JUDGE_MAX_TOKENS,
        **env_kwargs,
    )
    responder = responder_model(
        args.responder,
        base_url=args.responder_base_url,
        disable_thinking=args.responder_disable_thinking,
        max_tokens=args.responder_max_tokens,
    )
    if args.responder_disable_thinking:
        append_no_think(responder)

    prepare_run_directory(out, settings, args.resume)
    principle_ids = [p.id for p in role_spec.principles]
    random.seed(args.seed)
    last, state = (resume_state(out, role_spec, taxonomy, seeds, principle_ids)
                   if args.resume else (0, None))
    if last >= args.rounds:
        logger.info("All %d rounds are already complete.", args.rounds)
        return
    # Checkpoints store the weights used for the completed round; recompute
    # the schedule for the next round instead of restoring them.
    config.rebalance(last + 1, args.rounds)

    def prepared(number: int, data: dict) -> None:
        backend.check_round(number, [Scenario(**s) for s in data["round_scenarios"]])
        path = out / f"prepared_{number}.json"
        if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(data)):
            raise SystemExit(f"{path} differs from the round being resumed")
        _write_json(path, data)

    def checkpoint(number: int, data: dict) -> None:
        _write_json(out / f"round_{number}.json",
                    {**data, "frame_entropy": _frame_entropy(data.get("sample_payloads", []))})

    report = await run_campaign(
        role_spec=role_spec,
        taxonomy=taxonomy,
        seed_scenarios=seeds,
        config=config,
        planner_model=env,
        monitor_model=env,
        responder_model=responder,
        judge_model=judge,
        user_model=env,
        num_rounds=args.rounds,
        start_round=last + 1,
        initial_state=state,
        target_principle_ids=principle_ids,
        rollout=backend,
        round_prepared_callback=prepared,
        round_callback=checkpoint,
        checkpoint_errors_fatal=True,
    )
    _write_json(out / "campaign.json", {
        "role_id": role_spec.id,
        "settings": settings,
        "models": {"responder": model_name(responder), "environment": model_name(env)},
        **asdict(report),
    })
    print(_summary(report, out))
