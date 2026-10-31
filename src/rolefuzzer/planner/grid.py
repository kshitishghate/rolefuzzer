"""Coverage grid, feasibility tables and per-scenario outcome tracking."""

from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from rolefuzzer.spec import (
    DEVIATION_STATUSES,
    Scenario,
)

logger = logging.getLogger(__name__)


class FeasibilityTable:
    """Beta feasibility with hierarchical pooling across principles.

    Maintains two count tables:
      • cell-specific:  (role, mode, strategy, principle) -> [attempts, successes]
      • pooled:         (role, mode, strategy)            -> [attempts, successes]

    Score is an empirical-Bayes shrinkage between the two:

        score(cell) = (1 - λ) · p_cell + λ · p_pool
        λ = κ / (n_cell_attempts + κ)

    Early in a campaign λ → 1 so the planner uses the pooled estimate
    (converges fast,
    pooling evidence across principles). As cell-specific evidence
    accumulates, λ → 0 and the cell estimate dominates — capturing
    cases where some principles are structurally harder to elicit in a
    given (mode, strategy).

    Reference: standard hierarchical Bayesian shrinkage (Gelman et al.).
    """

    def __init__(
        self,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        kappa: float = 5.0,
    ) -> None:
        self._alpha_prior = alpha_prior
        self._beta_prior = beta_prior
        self._kappa = kappa
        # Cell-specific counts: (role, mode, strategy, principle) -> [attempts, successes]
        self._counts: dict[tuple[str, str, str, str], list[int]] = defaultdict(lambda: [0, 0])
        # Pooled counts across principles: (role, mode, strategy) -> [attempts, successes]
        self._pooled: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])

    def _beta_mean(self, attempts: int, successes: int) -> float:
        a, b = self._alpha_prior, self._beta_prior
        return (successes + a) / (attempts + a + b)

    def record_attempt(
        self, role_id: str, mode_id: str, strategy_id: str, principle_id: str = ""
    ) -> None:
        """Record that the pipeline attempted to hit this cell."""
        self._counts[(role_id, mode_id, strategy_id, principle_id)][0] += 1
        self._pooled[(role_id, mode_id, strategy_id)][0] += 1

    def record_success(
        self, role_id: str, mode_id: str, strategy_id: str, principle_id: str = ""
    ) -> None:
        """Record that the pipeline successfully reached this cell."""
        self._counts[(role_id, mode_id, strategy_id, principle_id)][1] += 1
        self._pooled[(role_id, mode_id, strategy_id)][1] += 1

    def attempts(
        self, role_id: str, mode_id: str, strategy_id: str, principle_id: str = ""
    ) -> int:
        """Return cell-specific attempt count for reachability decisions."""
        attempts, _ = self._counts.get(
            (role_id, mode_id, strategy_id, principle_id), [0, 0]
        )
        return attempts

    def score(
        self, role_id: str, mode_id: str, strategy_id: str, principle_id: str = ""
    ) -> float:
        """Return shrinkage score combining cell-specific and pooled estimates."""
        cell_a, cell_s = self._counts.get((role_id, mode_id, strategy_id, principle_id), [0, 0])
        pool_a, pool_s = self._pooled.get((role_id, mode_id, strategy_id), [0, 0])
        p_cell = self._beta_mean(cell_a, cell_s)
        p_pool = self._beta_mean(pool_a, pool_s)
        lam = self._kappa / (cell_a + self._kappa)
        return (1.0 - lam) * p_cell + lam * p_pool

    def to_dict(self) -> dict:
        """Serialize for JSON output. Exposes both cell and pooled scores so
        shrinkage behavior is auditable post-hoc."""
        result: dict = {}
        for (role_id, mode_id, strategy_id, principle_id), (attempts, successes) in self._counts.items():
            key_str = (
                f"{role_id}/{mode_id}/{strategy_id}/{principle_id}"
                if principle_id else f"{role_id}/{mode_id}/{strategy_id}"
            )
            pool_a, pool_s = self._pooled.get((role_id, mode_id, strategy_id), [0, 0])
            result[key_str] = {
                "attempts": attempts,
                "successes": successes,
                "cell_score": self._beta_mean(attempts, successes),
                "pooled_attempts": pool_a,
                "pooled_successes": pool_s,
                "pooled_score": self._beta_mean(pool_a, pool_s),
                "score": self.score(role_id, mode_id, strategy_id, principle_id),
            }
        return result


# ---------------------------------------------------------------------------
# Quality-adjusted feasibility
# ---------------------------------------------------------------------------


# Default backoff parameters for the exhaustion penalty. Counter is keyed
# by (role, mode, strategy, principle) and resets at campaign start
#, not at round boundary. The defaults below mean: after 3 distinct
# scenarios on a cell exhaust the monitor (`EXHAUSTION_BACKOFF_K`),
# subtract a penalty term from that cell's priority. After 6 distinct
# exhaustions (`EXHAUSTION_HARD_CAP`), schedule the cell last: after every
# other cell, so it is used only when no other cell remains.
EXHAUSTION_BACKOFF_K: int = 3
EXHAUSTION_BACKOFF_PENALTY: float = 1.0
EXHAUSTION_HARD_CAP: int = 6


class QualityFeasibilityTable:
    """Per-cell quality-adjusted feasibility.

    Sibling table to :class:`FeasibilityTable`. While that table
    tracks whether scenarios *hit* the target cell at all, this one
    tracks how *usable* the runs on that cell were:

    - ``attempted``: scenarios scheduled on this cell.
    - ``realized``: runs that produced any monitor-approved user turn.
    - ``monitor_approved``: monitor-approved turn count.
    - ``exhaustion_count``: scenarios that ended with monitor exhaustion.
    - ``block_reasons``: per-cell counter of rejected-turn flags and of
      invalid-sample reasons (prefixed ``invalid:``).

    The allocator reads only the distinct-exhaustion counts (for exhaustion
    backoff); the rest is reported for diagnostics.
    """

    def __init__(self) -> None:
        self._cells: dict[tuple[str, str, str, str], dict[str, int]] = (
            defaultdict(lambda: {
                "attempted": 0,
                "realized": 0,
                "monitor_approved": 0,
                "exhaustion_count": 0,
            })
        )
        self._block_reasons: dict[tuple[str, str, str, str], Counter] = (
            defaultdict(Counter)
        )
        # Distinct scenario IDs that exhausted on each cell.
        self._exhausted_scenarios: dict[tuple[str, str, str, str], set[str]] = (
            defaultdict(set)
        )

    def record_attempt(self, key: tuple[str, str, str, str]) -> None:
        self._cells[key]["attempted"] += 1

    def record_realized(self, key: tuple[str, str, str, str]) -> None:
        self._cells[key]["realized"] += 1

    def record_monitor_approved(self, key: tuple[str, str, str, str], n: int = 1) -> None:
        self._cells[key]["monitor_approved"] += n

    def record_exhaustion(
        self,
        key: tuple[str, str, str, str],
        scenario_id: str = "",
    ) -> None:
        self._cells[key]["exhaustion_count"] += 1
        if scenario_id:
            self._exhausted_scenarios[key].add(scenario_id)

    def record_block_reason(self, key: tuple[str, str, str, str], flag: str) -> None:
        self._block_reasons[key][flag] += 1

    def distinct_exhaustions(self, key: tuple[str, str, str, str]) -> int:
        return len(self._exhausted_scenarios.get(key, ()))

    def totals(self) -> dict[str, int]:
        """Campaign-wide attempted, realized and monitor-approved counts."""
        return {
            name: sum(counts[name] for counts in self._cells.values())
            for name in ("attempted", "realized", "monitor_approved")
        }

    def to_dict(self) -> dict:
        result: dict = {}
        for key, counts in self._cells.items():
            role_spec, mode, strategy, principle = key
            key_str = f"{role_spec}/{mode}/{strategy}/{principle}"
            result[key_str] = dict(counts)
            br = self._block_reasons.get(key)
            if br:
                result[key_str]["block_reasons"] = dict(br)
            result[key_str]["distinct_exhausted_scenarios"] = (
                self.distinct_exhaustions(key)
            )
        return result


# ---------------------------------------------------------------------------
# Coverage Grid (3D: mode_14 × strategy_56 × principle)
# ---------------------------------------------------------------------------


@dataclass
class CoverageGridCell:
    """Data for a single (mode_14, strategy_56, principle) cell."""

    count: int = 0
    """Number of turn-level observations (one per monitor check)."""
    scenario_ids: set[str] = field(default_factory=set)
    """Unique scenario IDs that contributed observations.  ``n_conversations``
    is ``len(scenario_ids)`` — use this for n_min thresholds, not ``count``."""
    violation_scenario_ids: set[str] = field(default_factory=set)
    """Unique scenario IDs whose conversation contained at least one
    violation in this cell.  ``n_violation_conversations`` is
    ``len(violation_scenario_ids)``.  Used by
    ``violation_rate_per_conversation`` for reporting at
    independent-conversation granularity (Wilson CIs, etc.)."""
    violation_statuses: list[str] = field(default_factory=list)
    trigger_distribution: dict[str, int] = field(default_factory=dict)
    best_scenario_id: str | None = None
    best_quality_score: float = 0.0

    @property
    def n_conversations(self) -> int:
        """Number of unique conversations that contributed to this cell."""
        return len(self.scenario_ids)

    @property
    def n_violation_conversations(self) -> int:
        """Conversations that produced at least one violation in this cell."""
        return len(self.violation_scenario_ids)

    @property
    def violation_rate(self) -> float | None:
        """Per-turn violation rate.

        Fraction of recorded turn-observations whose judge status is a
        deviation (``partial_deviation`` or ``clear_breach``). ``None`` if
        the cell has no judged turns.

        **Note**: turn observations within the same conversation are not
        independent — long conversations dominate the denominator and
        bias the rate.  Use ``violation_rate_per_conversation`` for
        cross-cell comparisons, n_min_est-thresholded reporting, or
        statistical claims; this turn-level rate is retained as the
        signal that drives the priority function (where the within-
        conversation correlation does not matter).
        """
        if not self.violation_statuses:
            return None
        violated = sum(
            1 for s in self.violation_statuses if s in DEVIATION_STATUSES
        )
        return violated / len(self.violation_statuses)

    @property
    def violation_rate_per_conversation(self) -> float | None:
        """Per-conversation violation rate at independent-sample granularity.

        Fraction of unique conversations contributing to this cell that
        produced at least one violation in any turn.  ``None`` if no
        conversations have been recorded.

        This is the rate to report in cell-level tables
        and Wilson confidence intervals — its denominator
        (``n_conversations``) is independent samples, so 1/1 = 100% on
        a single conversation does not inflate the cell's apparent
        violation rate the way a single 8-turn conversation can on
        ``violation_rate`` (where 8 violation observations from one
        conversation would read as a stable 100% turn-rate).
        """
        n_conv = self.n_conversations
        if n_conv == 0:
            return None
        return self.n_violation_conversations / n_conv


class CoverageGrid:
    """3D coverage tracker keyed on (mode_id, strategy_id, principle_id).

    Cells are (mode_14 × strategy_56 × principle): 4 instrumental modes ×
    4 active relational strategies × K principles (64 cells for K = 4).
    """

    def __init__(
        self,
        mode_ids: list[str],
        strategy_ids: list[str],
        principle_ids: list[str],
    ) -> None:
        self._mode_ids = list(mode_ids)
        self._strategy_ids = list(strategy_ids)
        self._principle_ids = list(principle_ids)
        self._grid: dict[tuple[str, str, str], CoverageGridCell] = defaultdict(CoverageGridCell)

    @property
    def mode_ids(self) -> tuple[str, ...]:
        return tuple(self._mode_ids)

    @property
    def strategy_ids(self) -> tuple[str, ...]:
        return tuple(self._strategy_ids)

    @property
    def principle_ids(self) -> tuple[str, ...]:
        return tuple(self._principle_ids)

    def materialized_cells(self) -> list[tuple[tuple[str, str, str], CoverageGridCell]]:
        """Every cell looked up or recorded so far, in first-access order."""
        return list(self._grid.items())

    def get_cell(
        self, mode_id: str, strategy_id: str, principle_id: str
    ) -> CoverageGridCell:
        """Return the cell for a (mode, strategy, principle) triple."""
        return self._grid[(mode_id, strategy_id, principle_id)]

    def cells_with_violations(
        self,
    ) -> list[tuple[str, str, str, CoverageGridCell]]:
        """Return cells that have at least one violation observation."""
        results: list[tuple[str, str, str, CoverageGridCell]] = []
        for m in self._mode_ids:
            for s in self._strategy_ids:
                for p in self._principle_ids:
                    cell = self._grid[(m, s, p)]
                    if (cell.violation_rate or 0) > 0:
                        results.append((m, s, p, cell))
        return results

    def record(
        self,
        mode_id: str,
        strategy_id: str,
        principle_id: str,
        violation_status: str | None = None,
        trigger: str | None = None,
        scenario_id: str | None = None,
        quality_score: float | None = None,
    ) -> None:
        """Record an observation for a (mode, strategy, principle) cell."""
        key = (mode_id, strategy_id, principle_id)
        cell = self._grid[key]
        cell.count += 1
        if scenario_id is not None:
            cell.scenario_ids.add(scenario_id)
        if violation_status is not None:
            cell.violation_statuses.append(violation_status)
            if scenario_id is not None and violation_status in DEVIATION_STATUSES:
                cell.violation_scenario_ids.add(scenario_id)
        if trigger is not None:
            cell.trigger_distribution[trigger] = cell.trigger_distribution.get(trigger, 0) + 1
        if quality_score is not None and scenario_id is not None:
            if quality_score > cell.best_quality_score:
                cell.best_quality_score = quality_score
                cell.best_scenario_id = scenario_id

    def coverage_ratio(self) -> float:
        """Fraction of cells with at least one observation."""
        total = len(self._mode_ids) * len(self._strategy_ids) * len(self._principle_ids)
        if total == 0:
            return 0.0
        covered = sum(
            1
            for m in self._mode_ids
            for s in self._strategy_ids
            for p in self._principle_ids
            if self._grid[(m, s, p)].count > 0
        )
        return covered / total

    def uncovered_cells(self) -> list[tuple[str, str, str]]:
        """Return list of (mode_id, strategy_id, principle_id) with zero observations."""
        return [
            (m, s, p)
            for m in self._mode_ids
            for s in self._strategy_ids
            for p in self._principle_ids
            if self._grid[(m, s, p)].count == 0
        ]

    def grid_neighbors(
        self, mode_id: str, strategy_id: str, principle_id: str
    ) -> list[tuple[str, str, str]]:
        """Return cells adjacent along one axis (differ in exactly one dimension)."""
        neighbors: list[tuple[str, str, str]] = []
        for m in self._mode_ids:
            for s in self._strategy_ids:
                for p in self._principle_ids:
                    diffs = (m != mode_id) + (s != strategy_id) + (p != principle_id)
                    if diffs == 1:
                        neighbors.append((m, s, p))
        return neighbors

    def prioritized_cells(
        self,
        n: int,
        feasibility_table: FeasibilityTable | None = None,
        role_id: str = "",
        w_cov: float = 1.0,
        w_v: float = 1.0,
        quality_feasibility: "QualityFeasibilityTable | None" = None,
        use_exhaustion_backoff: bool = False,
        n_min_breadth: int = 0,
        breadth_feasibility_threshold: float = 0.2,
        breadth_min_attempts_for_infeasible: int = 3,
    ) -> list[tuple[str, str, str]]:
        """Return the *n* highest-priority cells for exploration.

        Returns ``min(n, number of cells)`` cells. With
        ``use_exhaustion_backoff``, cells at ``EXHAUSTION_HARD_CAP`` distinct
        exhaustions are scheduled last: they follow every other cell
        (backed-off cells included), so they fill the list only when no
        other cell remains.

        Priority formula:
            priority(cell) = (w_cov / (n_conversations + 1)
                              + w_v * max_neighbor_violation)
                             * feasibility
                             + jitter

        A small random jitter breaks ties so that equal-priority cells
        (e.g. all empty cells on round 1) are explored in random order
        rather than always starting from mode 1.

        when ``n_min_breadth > 0`` and any provisionally reachable
        cell has fewer than ``n_min_breadth`` observations, the
        violation-weighting term is zeroed so budget flows to
        under-covered cells first. Cells with enough attempts and low
        generic feasibility are treated as unreachable for this release
        condition; no role/principle IDs are special-cased.
        """
        # Breadth-first gating: check whether all reachable cells have
        # crossed the minimum observation threshold. Untried cells remain
        # provisionally reachable; a cell is excluded only after the
        # generic feasibility table has enough attempts and a low score.
        if n_min_breadth > 0:
            def _breadth_satisfied(mm: str, ss: str, pp: str) -> bool:
                if self._grid[(mm, ss, pp)].n_conversations >= n_min_breadth:
                    return True
                if feasibility_table is None or not role_id:
                    return False
                attempts = feasibility_table.attempts(role_id, mm, ss, pp)
                if attempts < breadth_min_attempts_for_infeasible:
                    return False
                return (
                    feasibility_table.score(role_id, mm, ss, pp)
                    < breadth_feasibility_threshold
                )

            breadth_reached = all(
                _breadth_satisfied(mm, ss, pp)
                for mm in self._mode_ids
                for ss in self._strategy_ids
                for pp in self._principle_ids
            )
            effective_w_v = w_v if breadth_reached else 0.0
        else:
            effective_w_v = w_v

        scored: list[tuple[bool, float, str, str, str]] = []
        for m in self._mode_ids:
            for s in self._strategy_ids:
                for p in self._principle_ids:
                    cell = self._grid[(m, s, p)]
                    cov_term = w_cov / (cell.n_conversations + 1)

                    # Max neighbor violation rate
                    max_neighbor_v = 0.0
                    for nm, ns, np_ in self.grid_neighbors(m, s, p):
                        ncell = self._grid[(nm, ns, np_)]
                        rate = ncell.violation_rate
                        if rate is not None and rate > max_neighbor_v:
                            max_neighbor_v = rate
                    v_term = effective_w_v * max_neighbor_v

                    feasibility = 1.0
                    if feasibility_table is not None and role_id:
                        feasibility = feasibility_table.score(role_id, m, s, p)

                    priority = (cov_term + v_term) * feasibility

                    # Exhaustion backoff: shed priority once K distinct
                    # scenarios on this cell have exhausted; at the hard cap
                    # the cell is sorted after every uncapped cell.
                    capped = False
                    if use_exhaustion_backoff and quality_feasibility is not None and role_id:
                        qkey = (role_id, m, s, p)
                        distinct = quality_feasibility.distinct_exhaustions(qkey)
                        capped = distinct >= EXHAUSTION_HARD_CAP
                        if distinct >= EXHAUSTION_BACKOFF_K:
                            priority -= EXHAUSTION_BACKOFF_PENALTY * (
                                distinct - EXHAUSTION_BACKOFF_K + 1
                            )

                    priority = priority + random.uniform(0, 0.01)
                    scored.append((capped, priority, m, s, p))

        scored.sort(key=lambda x: (not x[0], x[1]), reverse=True)
        return [(m, s, p) for _, _, m, s, p in scored[:n]]

    def all_cells(self) -> list[tuple[str, str, str]]:
        """Return every (mode_id, strategy_id, principle_id) triple in the grid."""
        return [
            (m, s, p)
            for m in self._mode_ids
            for s in self._strategy_ids
            for p in self._principle_ids
        ]

    def random_cells(
        self, n: int, rng: "random.Random | None" = None
    ) -> list[tuple[str, str, str]]:
        """Return *n* cells sampled uniformly at random from the full grid.

        Ablation targeting arm: isolates coverage-directed
        cell selection from targeting in general by picking cells without
        regard to how under-covered they are. Samples without replacement
        when ``n`` does not exceed the grid size, else with replacement.
        """
        r = rng or random
        cells = self.all_cells()
        if not cells:
            return []
        if n <= len(cells):
            return r.sample(cells, n)
        return [r.choice(cells) for _ in range(n)]

    def render_coverage_context(self) -> str:
        """Render coverage state as text for planner prompts."""
        total = len(self._mode_ids) * len(self._strategy_ids) * len(self._principle_ids)
        covered = sum(
            1
            for m in self._mode_ids
            for s in self._strategy_ids
            for p in self._principle_ids
            if self._grid[(m, s, p)].count > 0
        )
        lines: list[str] = [
            "COVERAGE STATUS:",
            f"- Overall: {covered}/{total} cells covered ({self.coverage_ratio() * 100:.1f}%)",
        ]

        # Per-mode coverage
        lines.append("- Coverage by mode:")
        for m in self._mode_ids:
            m_total = len(self._strategy_ids) * len(self._principle_ids)
            m_covered = sum(
                1 for s in self._strategy_ids for p in self._principle_ids
                if self._grid[(m, s, p)].count > 0
            )
            pct = 100 * m_covered / m_total if m_total else 0
            lines.append(f"    Mode {m}: {m_covered}/{m_total} ({pct:.0f}%)")

        # Violation hotspots
        hotspots = self._get_hotspots(5)
        if hotspots:
            lines.append("- Violation hotspots:")
            for m, s, p, rate, count in hotspots:
                lines.append(
                    f"    ({m}, {s}, {p}): {rate * 100:.0f}% violation ({count} obs)"
                )

        return "\n".join(lines)

    def _get_hotspots(
        self, top_n: int
    ) -> list[tuple[str, str, str, float, int]]:
        """Return top cells by violation rate (ties broken by count)."""
        hotspots: list[tuple[str, str, str, float, int]] = []
        for m in self._mode_ids:
            for s in self._strategy_ids:
                for p in self._principle_ids:
                    cell = self._grid[(m, s, p)]
                    rate = cell.violation_rate
                    if rate is not None and rate > 0:
                        hotspots.append((m, s, p, rate, cell.count))
        hotspots.sort(key=lambda x: (x[3], x[4]), reverse=True)
        return hotspots[:top_n]

    def to_dict(self) -> dict:
        """Serialize for JSON output."""
        result: dict = {}
        for m in self._mode_ids:
            for s in self._strategy_ids:
                for p in self._principle_ids:
                    cell = self._grid[(m, s, p)]
                    if cell.count > 0:
                        key = f"{m}/{s}/{p}"
                        result[key] = {
                            "count": cell.count,
                            "n_conversations": cell.n_conversations,
                            "n_violation_conversations": cell.n_violation_conversations,
                            "violation_rate": cell.violation_rate,
                            "violation_rate_per_conversation": (
                                cell.violation_rate_per_conversation
                            ),
                            "trigger_distribution": cell.trigger_distribution,
                            "best_scenario_id": cell.best_scenario_id,
                        }
        return result


# ---------------------------------------------------------------------------
# Scenario Outcome Tracking
# ---------------------------------------------------------------------------


@dataclass
class ScenarioOutcome:
    """Tracks what a scenario actually produced across runs."""

    scenario_id: str
    scenario_name: str
    target_mode_id: str
    target_principle_id: str
    runs: int = 0
    violations: int = 0
    observed_behavior_ids: list[str] = field(default_factory=list)
    turns_used_per_run: list[int] = field(default_factory=list)
    early_terminations: int = 0

    @property
    def violation_rate(self) -> float:
        return self.violations / self.runs if self.runs > 0 else 0.0

    @property
    def avg_turns_used(self) -> float:
        """Average number of substantive turns before termination."""
        if not self.turns_used_per_run:
            return 0.0
        return sum(self.turns_used_per_run) / len(self.turns_used_per_run)

    @property
    def early_termination_rate(self) -> float:
        return self.early_terminations / self.runs if self.runs > 0 else 0.0

    @property
    def behavior_profile(self) -> dict[str, int]:
        """Frequency of each observed TUNA behavior across runs."""
        counts: dict[str, int] = defaultdict(int)
        for bid in self.observed_behavior_ids:
            counts[bid] += 1
        return dict(counts)


class ScenarioOutcomeTracker:
    """Tracks scenario-level outcomes across campaign rounds."""

    def __init__(self) -> None:
        self._outcomes: dict[str, ScenarioOutcome] = {}

    def record(
        self,
        scenario: Scenario,
        target_mode_id: str,
        had_violation: bool,
        observed_behavior_ids: list[str] | None = None,
        turns_used: int | None = None,
        early_terminated: bool = False,
    ) -> None:
        """Record the outcome of a single conversation run."""
        if scenario.id not in self._outcomes:
            self._outcomes[scenario.id] = ScenarioOutcome(
                scenario_id=scenario.id,
                scenario_name=scenario.name,
                target_mode_id=target_mode_id,
                target_principle_id=scenario.target_principle_id,
            )
        outcome = self._outcomes[scenario.id]
        outcome.runs += 1
        if had_violation:
            outcome.violations += 1
        if observed_behavior_ids:
            outcome.observed_behavior_ids.extend(observed_behavior_ids)
        if turns_used is not None:
            outcome.turns_used_per_run.append(turns_used)
        if early_terminated:
            outcome.early_terminations += 1

    def violation_producing_scenarios(
        self, min_runs: int = 1
    ) -> list[ScenarioOutcome]:
        """Return scenarios that produced at least one violation, sorted by rate."""
        results = [
            o for o in self._outcomes.values()
            if o.violations > 0 and o.runs >= min_runs
        ]
        results.sort(key=lambda o: o.violation_rate, reverse=True)
        return results

    def to_dict(self) -> dict:
        """Serialize for JSON output."""
        return {
            sid: {
                "scenario_name": o.scenario_name,
                "target_mode_id": o.target_mode_id,
                "target_principle_id": o.target_principle_id,
                "runs": o.runs,
                "violations": o.violations,
                "violation_rate": o.violation_rate,
                "avg_turns_used": o.avg_turns_used,
                "early_terminations": o.early_terminations,
                "early_termination_rate": o.early_termination_rate,
                "behavior_profile": o.behavior_profile,
            }
            for sid, o in self._outcomes.items()
        }
