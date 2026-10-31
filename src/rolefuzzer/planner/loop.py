"""The campaign loop and the conversational rollout."""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Awaitable

from inspect_ai.model import (
    Model,
)

from rolefuzzer.planner.assess import assess_round, extract_sample_payload
from rolefuzzer.planner.config import CampaignConfig, CampaignReport, RoundReport
from rolefuzzer.planner.diagnostics import extract_cross_scenario_patterns
from rolefuzzer.planner.generation import generate_scenario_for_cell, generate_scenario_no_grid
from rolefuzzer.planner.grid import (
    CoverageGrid,
    FeasibilityTable,
    QualityFeasibilityTable,
    ScenarioOutcomeTracker,
)
from rolefuzzer.planner.mutation import (
    MUTATION_OPERATORS,
    _has_complete_target_cell,
    mutate_scenario,
    mutation_depth,
)
from rolefuzzer.spec import (
    MutationType,
    RoleSpec,
    Scenario,
)
from rolefuzzer.taxonomy import TunaTaxonomy

logger = logging.getLogger(__name__)


async def chat_rollout(
    *,
    role_spec: RoleSpec,
    scenarios: list[Scenario],
    config: "CampaignConfig",
    user_model: str | Model,
    judge_model: str | Model,
    monitor_model: str | Model | None,
    responder_model: str | Model,
    taxonomy_summary: str = "",
) -> list[dict]:
    """Default rollout: run scenarios as conversations through Inspect.

    The campaign loop takes the rollout as a parameter so other interaction
    media (the web-agent backend) plug in without branching. A sample that
    raised an error is kept, marked ``invalid_sample``, so every scenario
    yields a payload.
    """
    from inspect_ai import Task
    from inspect_ai import eval_async as inspect_eval

    from rolefuzzer.conversation import role_conversation
    from rolefuzzer.dataset import role_dataset_from_scenarios
    from rolefuzzer.scorer import principle_violation_scorer

    dataset = role_dataset_from_scenarios(
        role_spec=role_spec,
        scenarios=scenarios,
        iterations=1,
    )

    # One sample's error must not stop the others; it is recorded below.
    eval_kwargs: dict[str, Any] = {"fail_on_error": False}
    if config.max_concurrent is not None:
        eval_kwargs["max_samples"] = config.max_concurrent
    if config.log_dir is not None:
        eval_kwargs["log_dir"] = config.log_dir

    results = await inspect_eval(
        tasks=[
            Task(
                dataset=dataset,
                solver=role_conversation(
                    user_model=user_model,
                    num_turns=config.num_turns,
                    judge_model=judge_model,
                    monitor_model=monitor_model,
                    taxonomy_summary=taxonomy_summary,
                ),
                scorer=principle_violation_scorer(),
            )
        ],
        model=responder_model,
        **eval_kwargs,
    )

    payloads: list[dict] = []
    for log in results:
        for sample in log.samples or []:
            payload = extract_sample_payload(sample)
            if sample.error is not None:
                payload["metadata"]["invalid_sample"] = True
                payload["metadata"]["invalid_reason"] = (
                    f"sample_error: {sample.error.message}"[:300]
                )
            payloads.append(payload)
    if len(payloads) != len(dataset):
        statuses = ", ".join(sorted({log.status for log in results}))
        raise RuntimeError(
            f"Evaluation returned {len(payloads)} of {len(dataset)} samples "
            f"(log status: {statuses})"
        )
    return payloads


# ---------------------------------------------------------------------------
# Campaign state
# ---------------------------------------------------------------------------


@dataclass
class _CampaignState:
    """What a campaign carries from one round to the next."""

    grid: CoverageGrid
    feasibility: FeasibilityTable
    quality_feasibility: QualityFeasibilityTable
    outcome_tracker: ScenarioOutcomeTracker
    scenario_lookup: dict[str, Scenario] = field(default_factory=dict)
    diagnostic_by_scenario: dict[str, str] = field(default_factory=dict)
    round_reports: list[RoundReport] = field(default_factory=list)
    total_conversations: int = 0
    all_consistency_checks: list[dict] = field(default_factory=list)

    @classmethod
    def start(
        cls, grid_axes: tuple[list[str], list[str], list[str]], initial_state: dict | None,
    ) -> "_CampaignState":
        """A fresh state, or the one a resumed run's checkpoints rebuilt."""
        if initial_state is None:
            return cls(
                grid=CoverageGrid(*grid_axes),
                feasibility=FeasibilityTable(),
                quality_feasibility=QualityFeasibilityTable(),
                outcome_tracker=ScenarioOutcomeTracker(),
            )
        return cls(
            grid=initial_state.get("grid") or CoverageGrid(*grid_axes),
            feasibility=initial_state.get("feasibility") or FeasibilityTable(),
            quality_feasibility=(
                initial_state.get("quality_feasibility") or QualityFeasibilityTable()
            ),
            outcome_tracker=initial_state.get("outcome_tracker") or ScenarioOutcomeTracker(),
            scenario_lookup=dict(initial_state.get("scenario_lookup") or {}),
            diagnostic_by_scenario=dict(initial_state.get("diagnostic_by_scenario") or {}),
            round_reports=list(initial_state.get("round_reports") or []),
            total_conversations=int(initial_state.get("total_conversations") or 0),
            all_consistency_checks=list(initial_state.get("all_consistency_checks") or []),
        )

    def register(self, scenario: Scenario) -> None:
        self.scenario_lookup[scenario.id] = scenario


class _SeedRotation:
    """Round-robin seed sampling.

    Each seed has a use counter that persists across rounds. One call draws
    without replacement, preferring the least-used seeds (ties broken by the
    global ``random`` state). Min-use balancing is self-resetting, so no
    cooldown is needed.
    """

    def __init__(self, seeds: list[Scenario], use_counts: dict[str, int] | None = None) -> None:
        self.seeds = seeds
        self.use_counts: dict[str, int] = {s.id: 0 for s in seeds}
        self.use_counts.update(use_counts or {})

    def sample(self, n: int, *, principle_id: str | None = None) -> list[Scenario]:
        """Pick *n* seeds, preferring lowest-use; no repeats within one call."""
        if not self.seeds or n <= 0:
            return []
        pool = [
            scenario for scenario in self.seeds
            if principle_id is None
            or scenario.target_principle_id == principle_id
        ]
        if not pool:
            return []
        random.shuffle(pool)  # break ties randomly
        pool.sort(key=lambda s: self.use_counts.get(s.id, 0))
        chosen = pool[: min(n, len(pool))]
        for s in chosen:
            self.use_counts[s.id] = self.use_counts.get(s.id, 0) + 1
        return chosen


@dataclass
class _Campaign:
    """The fixed inputs of a campaign plus its evolving state."""

    role_spec: RoleSpec
    taxonomy: TunaTaxonomy
    seed_scenarios: list[Scenario]
    config: CampaignConfig
    planner_model: str | Model
    state: _CampaignState
    seeds: _SeedRotation


# ---------------------------------------------------------------------------
# Scenario generation, one function per bucket
# ---------------------------------------------------------------------------


def _bucket_sizes(campaign: _Campaign, round_budget: int, round_num: int) -> dict[str, int]:
    """Split the round budget, moving mutation budget to coverage when no
    complete-cell elite exists to mutate."""
    buckets = campaign.config.bucket_sizes(round_budget)
    if buckets["mutation"] > 0:
        lookup = campaign.state.scenario_lookup
        has_complete_elite = any(
            cell.best_scenario_id
            and cell.best_scenario_id in lookup
            and _has_complete_target_cell(lookup[cell.best_scenario_id])
            for _, _, _, cell in campaign.state.grid.cells_with_violations()
        )
        if not has_complete_elite:
            logger.info(
                "Round %d: no complete-cell mutation elites available; "
                "reallocating mutation budget (%d) to coverage.",
                round_num,
                buckets["mutation"],
            )
            buckets["coverage"] += buckets["mutation"]
            buckets["mutation"] = 0
    return buckets


async def _no_grid_scenarios(
    campaign: _Campaign, round_budget: int, round_num: int,
) -> list[Scenario]:
    """The ``no-grid`` search mode: prompt-only generation with the grid bypassed.

    The round budget is spread across the role's principles and each
    scenario is generated from the role spec alone (no seed/coverage/mutation
    split, no cell targeting).
    """
    principles = list(campaign.role_spec.principles)

    # Roles whose scenarios execute in an environment need each scenario
    # bound to one; anchor to a seed of the same principle (least-used
    # first) for its environment facts only. Seeds without an execution
    # binding (conversational roles) are never used here.
    anchored = any(s.execution_binding for s in campaign.seed_scenarios or [])
    anchors = [
        (campaign.seeds.sample(
            1, principle_id=principles[i % len(principles)].id,
        ) or [None])[0] if anchored else None
        for i in range(round_budget)
    ]

    async def generate(idx: int) -> Scenario | None:
        principle = principles[idx % len(principles)]
        try:
            return await generate_scenario_no_grid(
                planner_model=campaign.planner_model,
                role_spec=campaign.role_spec,
                target_principle=principle,
                execution_anchor=anchors[idx],
            )
        except Exception as exc:
            logger.warning(
                "Round %d: no_grid generation failed (%s): %s",
                round_num, principle.id, exc,
            )
            return None

    results = await asyncio.gather(*[generate(i) for i in range(round_budget)])
    return [sc for sc in results if sc is not None]


def _seed_scenarios(campaign: _Campaign, n: int) -> list[Scenario]:
    """Seed bucket: the least-used seeds, unsteered, as a baseline.

    Tagged with ``seed_baseline=true`` so the solver skips monitor steering
    for unbiased control data. ``seed_use_count`` is logged so post-hoc
    analysis can verify the diversity guarantee.
    """
    tagged: list[Scenario] = []
    for s in campaign.seeds.sample(min(n, len(campaign.seed_scenarios))):
        tagged.append(s.model_copy(update={"metadata": {
            **s.metadata,
            "seed_baseline": "true",
            "seed_id": s.id,
            "frame_anchor_seed_id": s.id,
            "seed_use_count": str(campaign.seeds.use_counts.get(s.id, 0)),
        }}))
    return tagged


async def _coverage_scenarios(
    campaign: _Campaign, n: int, round_num: int,
) -> list[Scenario]:
    """Coverage bucket: new scenarios for under-covered cells."""
    config, state = campaign.config, campaign.state
    grid = state.grid
    if config.cell_selection == "random":
        # Random targeting holds the rest of the pipeline (generator, seed
        # anchor, mutation) fixed, isolating coverage-directed targeting
        # from targeting in general.
        priority_cells = grid.random_cells(n, rng=random)
    else:
        priority_cells = grid.prioritized_cells(
            n,
            feasibility_table=state.feasibility,
            role_id=campaign.role_spec.id,
            quality_feasibility=state.quality_feasibility,
            use_exhaustion_backoff=config.use_exhaustion_backoff,
            n_min_breadth=config.n_min_breadth,
        )
    coverage_context = grid.render_coverage_context()
    # Append cross-scenario patterns for rounds after the first
    if round_num > 1:
        patterns = extract_cross_scenario_patterns(grid)
        coverage_context = coverage_context + "\n\n" + patterns

    async def generate(m_id: str, s_id: str, p_id: str) -> Scenario | None:
        try:
            target_mode = campaign.taxonomy.get_mode(m_id)
            target_strategy = campaign.taxonomy.get_strategy(s_id)
            target_principle = campaign.role_spec.get_principle(p_id)
            diag = ""
            for nm, ns, np_ in grid.grid_neighbors(m_id, s_id, p_id):
                nb_id = grid.get_cell(nm, ns, np_).best_scenario_id
                if nb_id and nb_id in state.diagnostic_by_scenario:
                    diag = state.diagnostic_by_scenario[nb_id]
                    break
            # Anchor each coverage-bucket scenario to a seed so situational
            # variety comes from the seed corpus, not the planner's narrow
            # prior. Uses the same round-robin min-use balancing as the seed
            # bucket.
            anchor_seeds = campaign.seeds.sample(
                1,
                principle_id=(
                    p_id
                    if config.require_principle_matched_execution_anchor
                    else None
                ),
            )
            anchor_seed = anchor_seeds[0] if anchor_seeds else None
            scenario = await generate_scenario_for_cell(
                planner_model=campaign.planner_model,
                role_spec=campaign.role_spec,
                target_principle=target_principle,
                target_mode=target_mode,
                target_strategy=target_strategy,
                taxonomy=campaign.taxonomy,
                seed_scenario=anchor_seed,
                coverage_context=coverage_context,
                diagnostic_context=diag,
            )
            # Log seed lineage on the generated scenario so downstream
            # tooling (frame entropy, audit) can trace it. The frame anchor
            # is the seed chosen here, never the planner's own claim, so a
            # frame_swap mutation can move away from it; with no seed drawn
            # the scenario has no anchor ("").
            if anchor_seed is None:
                return scenario.model_copy(update={"metadata": {
                    **scenario.metadata, "frame_anchor_seed_id": "",
                }})
            return scenario.model_copy(update={"metadata": {
                **scenario.metadata,
                "seed_id": anchor_seed.id,
                "frame_anchor_seed_id": anchor_seed.id,
                "seed_use_count": str(campaign.seeds.use_counts.get(anchor_seed.id, 0)),
                "execution_anchor_principle_id": anchor_seed.target_principle_id,
            }})
        except Exception as exc:
            logger.warning(
                "Failed to generate scenario for cell (%s, %s, %s): %s",
                m_id, s_id, p_id, exc,
            )
            return None

    results = await asyncio.gather(*[generate(m, s, p) for m, s, p in priority_cells])
    return [scenario for scenario in results if scenario is not None]


async def _mutation_scenarios(
    campaign: _Campaign, n: int, round_num: int,
) -> list[Scenario]:
    """Mutation bucket: within-cell mutations of the cells' elite scenarios.

    Both operators keep the parent's cell; cross-cell exploration is the
    coverage bucket's job. Elite parents are drawn with multiplicity
    proportional to the number of conversations that contributed to the
    cell, so cells with better-estimated quality scores are preferred over
    single-shot lucky violators.
    """
    config, state = campaign.config, campaign.state
    elite_pool: list[Scenario] = []
    depth_capped = 0
    for _, _, _, cell in state.grid.cells_with_violations():
        if cell.best_scenario_id and cell.best_scenario_id in state.scenario_lookup:
            elite = state.scenario_lookup[cell.best_scenario_id]
            if not _has_complete_target_cell(elite):
                continue
            # Cap mutation lineage depth so the elite pool can't drift to
            # ever-deeper mutated descendants; deep chains concentrate the
            # budget on one synthetic lineage.
            depth = mutation_depth(elite, state.scenario_lookup)
            if depth >= config.max_mutation_depth:
                depth_capped += 1
                continue
            weight = max(1, cell.n_conversations)
            elite_pool.extend([elite] * weight)
    if depth_capped:
        logger.debug(
            "Round %d: excluded %d cell elite(s) at depth >= %d from mutation pool.",
            round_num, depth_capped, config.max_mutation_depth,
        )
    if not elite_pool:
        logger.info(
            "Round %d: no complete-cell mutation elites available "
            "(after depth cap %d); skipping mutation bucket instead "
            "of mutating blind.",
            round_num, config.max_mutation_depth,
        )
        return []

    async def mutate(base: Scenario, mut_type: MutationType) -> Scenario | None:
        try:
            return await mutate_scenario(
                planner_model=campaign.planner_model,
                base_scenario=base,
                mutation_type=mut_type,
                role_spec=campaign.role_spec,
                taxonomy=campaign.taxonomy,
                seed_pool=campaign.seed_scenarios,
                rng=random,
                diagnostic_context=state.diagnostic_by_scenario.get(base.id, ""),
                require_principle_matched_anchor=(
                    config.require_principle_matched_execution_anchor
                ),
            )
        except Exception as exc:
            logger.warning(
                "Failed to mutate %s with %s: %s",
                base.id, mut_type, exc,
            )
            return None

    # Round-robin across elites and operators so the mutation budget hits
    # each operator in turn.
    results = await asyncio.gather(*[
        mutate(elite_pool[i % len(elite_pool)], MUTATION_OPERATORS[i % len(MUTATION_OPERATORS)])
        for i in range(n)
    ])
    return [mutated for mutated in results if mutated is not None]


async def _round_scenarios(
    campaign: _Campaign, round_budget: int, round_num: int, restored: list[Scenario] | None,
) -> list[Scenario]:
    """This round's scenarios: restored from a checkpoint, or generated."""
    state = campaign.state
    if restored is not None:
        round_scenarios = list(restored)
        for s in round_scenarios:
            state.register(s)
        return round_scenarios
    if campaign.config.no_grid:
        round_scenarios = await _no_grid_scenarios(campaign, round_budget, round_num)
        for s in round_scenarios:
            state.register(s)
        return round_scenarios

    buckets = _bucket_sizes(campaign, round_budget, round_num)
    round_scenarios = _seed_scenarios(campaign, buckets["seed"])
    if buckets["coverage"] > 0:
        generated = await _coverage_scenarios(campaign, buckets["coverage"], round_num)
        for s in generated:
            state.register(s)
        round_scenarios += generated
    if buckets["mutation"] > 0:
        mutated = await _mutation_scenarios(campaign, buckets["mutation"], round_num)
        for s in mutated:
            state.register(s)
        round_scenarios += mutated
    return round_scenarios


# ---------------------------------------------------------------------------
# Assess, checkpoint and report
# ---------------------------------------------------------------------------


async def _assess_and_checkpoint(
    campaign: _Campaign,
    round_num: int,
    round_scenarios: list[Scenario],
    sample_payloads: list[dict],
    prev_coverage: float,
    prev_uncovered: int,
    round_callback: Any | None,
    checkpoint_errors_fatal: bool,
) -> float:
    """Fold a round's conversations into the campaign state, report and
    checkpoint it. Returns the coverage ratio after the round."""
    state, config = campaign.state, campaign.config
    conversations_this_round, round_checks = assess_round(
        role_spec=campaign.role_spec,
        sample_payloads=sample_payloads,
        grid=state.grid,
        feasibility=state.feasibility,
        outcome_tracker=state.outcome_tracker,
        scenario_lookup=state.scenario_lookup,
        diagnostic_by_scenario=state.diagnostic_by_scenario,
        quality_feasibility=state.quality_feasibility,
    )
    state.all_consistency_checks.extend(round_checks)
    state.total_conversations += conversations_this_round

    new_coverage = state.grid.coverage_ratio()
    new_uncovered = len(state.grid.uncovered_cells())
    state.round_reports.append(
        RoundReport(
            round_number=round_num,
            scenarios_generated=len(round_scenarios),
            conversations_run=conversations_this_round,
            coverage_ratio=new_coverage,
            new_cells_covered=prev_uncovered - new_uncovered,
        )
    )

    logger.info(
        "Round %d: coverage %.1f%% -> %.1f%% (+%d cells), %d conversations",
        round_num,
        prev_coverage * 100,
        new_coverage * 100,
        prev_uncovered - new_uncovered,
        conversations_this_round,
    )

    # Per-round callback (e.g. checkpoint to disk).
    if round_callback is not None:
        try:
            cb_result = round_callback(
                round_num,
                {
                    "round_number": round_num,
                    "sample_payloads": sample_payloads,
                    "round_scenarios": [s.model_dump() for s in round_scenarios],
                    "coverage_ratio": new_coverage,
                    "new_cells_covered": prev_uncovered - new_uncovered,
                    "consistency_checks": round_checks,
                    "quality_feasibility": state.quality_feasibility.to_dict(),
                    "alpha_seed": config.alpha_seed,
                    "alpha_coverage": config.alpha_coverage,
                    "alpha_mutation": config.alpha_mutation,
                    "seed_use_counts": dict(campaign.seeds.use_counts),
                    "rng_state": random.getstate(),
                },
            )
            if isinstance(cb_result, Awaitable):
                await cb_result
        except Exception as exc:
            if checkpoint_errors_fatal:
                raise
            logger.warning("Round callback failed (round %d): %s", round_num, exc)
    return new_coverage


def _monitor_exhaustion_exceeded(
    config: CampaignConfig, round_num: int, sample_payloads: list[dict],
) -> bool:
    """Whether the round's monitor-exhaustion rate calls for stopping.

    A ``monitor_exhausted`` termination means every revision attempt was
    rejected by the role gate or consistency monitor, usually a prompt or
    role-spec regression. Running further rounds in that state burns budget
    and pollutes mutation elites with malformed parents.
    """
    if config.monitor_exhaustion_abort_threshold is None or not sample_payloads:
        return False
    exhausted = sum(
        1 for p in sample_payloads
        if (p.get("metadata") or {}).get("termination_reason") == "monitor_exhausted"
    )
    exhaustion_rate = exhausted / len(sample_payloads)
    if exhaustion_rate <= config.monitor_exhaustion_abort_threshold:
        return False
    logger.warning(
        "Round %d monitor-exhaustion rate %.1f%% (%d/%d) exceeded "
        "abort threshold %.1f%%; stopping campaign so the run can "
        "be inspected before further budget is spent.",
        round_num,
        exhaustion_rate * 100,
        exhausted,
        len(sample_payloads),
        config.monitor_exhaustion_abort_threshold * 100,
    )
    return True


def _final_report(state: _CampaignState, stop_reason: str) -> CampaignReport:
    consistency_stats = {}
    if state.all_consistency_checks:
        total_checks = len(state.all_consistency_checks)
        approved = sum(1 for c in state.all_consistency_checks if c.get("approved", False))
        used_revision = sum(
            1 for c in state.all_consistency_checks if c.get("used_revision", False)
        )
        consistency_stats = {
            "total_checks": total_checks,
            "approval_rate": approved / total_checks if total_checks else 0,
            "revision_rate": used_revision / total_checks if total_checks else 0,
        }

    # Coverage totals: the raw attempted budget, what the classifier admitted
    # as realised, and what the monitor approved. Monitor approval is not a
    # post-hoc quality audit, so it is not reported as "usable".
    return CampaignReport(
        total_rounds=len(state.round_reports),
        total_conversations=state.total_conversations,
        final_coverage_ratio=state.grid.coverage_ratio(),
        round_reports=state.round_reports,
        coverage_data={
            "grid": state.grid.to_dict(),
            "feasibility": state.feasibility.to_dict(),
            "quality_feasibility": state.quality_feasibility.to_dict(),
            "scenario_outcomes": state.outcome_tracker.to_dict(),
            "coverage_totals": state.quality_feasibility.totals(),
        },
        consistency_stats=consistency_stats,
        stop_reason=stop_reason,
    )


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------


async def run_campaign(
    role_spec: RoleSpec,
    taxonomy: TunaTaxonomy,
    seed_scenarios: list[Scenario],
    config: CampaignConfig,
    planner_model: str | Model,
    monitor_model: str | Model | None,
    responder_model: str | Model,
    judge_model: str | Model,
    user_model: str | Model,
    num_rounds: int = 3,
    conversations_per_round: int | None = None,
    round_callback: Any | None = None,
    start_round: int = 1,
    initial_state: dict | None = None,
    target_principle_ids: list[str] | None = None,
    rollout: Any | None = None,
    round_prepared_callback: Any | None = None,
    checkpoint_errors_fatal: bool = False,
) -> CampaignReport:
    """Run a fuzzing campaign with three-bucket budget allocation.

    Each round:
    1. Split budget across seed / coverage / mutation buckets (time-annealed)
    2. Generate scenarios per bucket using CoverageGrid priorities
    3. Run conversations with monitor (classify + steer)
    4. Update CoverageGrid + FeasibilityTable from monitor verdicts
    5. Anneal bucket alphas for the next round (open-loop AFLGo-style)
    6. Check stopping rules (coverage threshold, budget exhausted)

    Args:
        role_spec: The role spec to test.
        taxonomy: Full TUNA taxonomy.
        seed_scenarios: Initial seed scenarios.
        config: Campaign configuration (budget, weights, thresholds).
        planner_model: Model for scenario generation.
        monitor_model: Model for consistency monitor.
        responder_model: Model under test.
        judge_model: Model for violation judging.
        user_model: Model for user simulation.
        num_rounds: Maximum number of rounds.
        conversations_per_round: Override per-round budget (default: config.round_budget).
        rollout: Async callable that turns a round's scenarios into
            `sample_payloads` dicts. Defaults to :func:`chat_rollout`, the
            Inspect conversational rollout. The web-agent backend passes its
            own. This parameterises the interaction medium, not the role.

    Returns:
        CampaignReport with coverage grid data and the reason the campaign
        stopped (see ``CampaignReport.stop_reason``).
    """
    from inspect_ai._eval.eval import display_type_initialized, init_display_type

    if not display_type_initialized():
        init_display_type("plain")

    # Build grid axes from taxonomy
    mode_ids = [m.id for m in taxonomy.instrumental_modes()]
    strategy_ids = [s.id for s in taxonomy.targetable_relational_strategies()]
    all_principle_ids = [p.id for p in role_spec.principles]
    # `target_principle_ids` restricts the GRID only — the planner still
    # sees `role.principles` (full list) when rendering prompts, and the
    # judge still evaluates every role principle per turn via
    # `role_dataset_from_scenarios`. Restricting the grid means the
    # coverage and mutation buckets only target cells in those principles,
    # while cross-principle breaches by the responder are still recorded
    # by the judge.
    if target_principle_ids:
        unknown = [p for p in target_principle_ids if p not in all_principle_ids]
        if unknown:
            raise ValueError(
                f"target_principle_ids contains principle(s) not in role "
                f"{role_spec.id!r}: {unknown}. Role principles: {all_principle_ids}"
            )
        principle_ids = list(target_principle_ids)
        logger.info(
            "Campaign target restricted to principle(s): %s "
            "(judge still evaluates all %d role principles).",
            principle_ids, len(all_principle_ids),
        )
    else:
        principle_ids = all_principle_ids

    state = _CampaignState.start((mode_ids, strategy_ids, principle_ids), initial_state)
    # Restore annealed alpha weights so the schedule isn't reset on resume.
    if initial_state is not None and "alpha_seed" in initial_state:
        config.alpha_seed = float(initial_state["alpha_seed"])
        config.alpha_coverage = float(initial_state["alpha_coverage"])
        config.alpha_mutation = float(initial_state["alpha_mutation"])

    round_budget = conversations_per_round or config.round_budget

    for s in seed_scenarios:
        state.register(s)

    seeds = _SeedRotation(seed_scenarios, (initial_state or {}).get("seed_use_counts"))
    if initial_state and initial_state.get("rng_state"):
        def _tuples(value):
            return tuple(_tuples(v) for v in value) if isinstance(value, list) else value
        random.setstate(_tuples(initial_state["rng_state"]))

    campaign = _Campaign(
        role_spec=role_spec,
        taxonomy=taxonomy,
        seed_scenarios=seed_scenarios,
        config=config,
        planner_model=planner_model,
        state=state,
        seeds=seeds,
    )
    rollout_fn = rollout or chat_rollout

    stop_reason = "rounds_complete"
    for round_num in range(start_round, num_rounds + 1):
        prev_coverage = state.grid.coverage_ratio()
        prev_uncovered = len(state.grid.uncovered_cells())

        restored = (
            (initial_state or {}).get("pending_round_scenarios")
            if round_num == start_round else None
        )
        round_scenarios = await _round_scenarios(campaign, round_budget, round_num, restored)

        # A round without scenarios (every generation failed) is still a
        # round: it is checkpointed with no conversations and counts toward
        # the schedule and the stopping rules.
        if not round_scenarios:
            logger.warning("Round %d: no scenarios available; recording an empty round.",
                           round_num)

        if round_prepared_callback is not None:
            prepared = round_prepared_callback(round_num, {
                "round_scenarios": [s.model_dump() for s in round_scenarios],
                "seed_use_counts": dict(seeds.use_counts),
                "rng_state": random.getstate(),
            })
            if isinstance(prepared, Awaitable):
                await prepared

        # --- Run: turn this round's scenarios into conversations ---
        # The rollout backend is injected so the campaign loop is agnostic to
        # the interaction medium. Everything after this point consumes
        # `sample_payloads` dicts and is identical for every rollout.
        sample_payloads = await rollout_fn(
            role_spec=role_spec,
            scenarios=round_scenarios,
            config=config,
            user_model=user_model,
            judge_model=judge_model,
            monitor_model=monitor_model,
            responder_model=responder_model,
            taxonomy_summary=taxonomy.render_modes_summary(),
        ) if round_scenarios else []

        new_coverage = await _assess_and_checkpoint(
            campaign, round_num, round_scenarios, sample_payloads,
            prev_coverage, prev_uncovered, round_callback, checkpoint_errors_fatal,
        )

        if _monitor_exhaustion_exceeded(config, round_num, sample_payloads):
            stop_reason = "monitor_exhaustion"
            break

        # Time-annealed rebalance (open-loop AFLGo-style schedule).
        # The next round uses the freshly interpolated alphas.
        config.rebalance(round_num=round_num + 1, total_rounds=num_rounds)

        # Stopping rule: coverage threshold reached. Checked before the
        # manual stop so a finished campaign is never recorded as paused.
        if new_coverage >= config.coverage_threshold:
            logger.info(
                "Coverage threshold %.1f%% reached, stopping campaign.",
                config.coverage_threshold * 100,
            )
            stop_reason = "coverage_threshold"
            break

        # Manual mid-campaign stop. Run was launched with num_rounds=N so
        # the alpha schedule was planned for N rounds, but the operator
        # only wants to execute through `stop_after_round`. Exit cleanly
        # so checkpoints are intact and the campaign can resume later.
        if (
            config.stop_after_round is not None
            and round_num >= config.stop_after_round
            and round_num < num_rounds
        ):
            logger.info(
                "Reached stop_after_round=%d (of planned num_rounds=%d); "
                "exiting cleanly. Resume from round %d to continue under "
                "the same alpha schedule.",
                config.stop_after_round,
                num_rounds,
                round_num + 1,
            )
            stop_reason = "stop_after_round"
            break

    return _final_report(state, stop_reason)
