"""Campaign configuration and reports."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Campaign Report
# ---------------------------------------------------------------------------


@dataclass
class RoundReport:
    """Report for a single campaign round."""

    round_number: int
    scenarios_generated: int
    conversations_run: int
    coverage_ratio: float
    new_cells_covered: int


@dataclass
class CampaignReport:
    """Final report for a full campaign."""

    total_rounds: int
    total_conversations: int
    final_coverage_ratio: float
    round_reports: list[RoundReport]
    coverage_data: dict
    consistency_stats: dict = field(default_factory=dict)
    # Why the campaign loop ended: "rounds_complete", "stop_after_round"
    # (paused; a resume continues it), "coverage_threshold" or
    # "monitor_exhaustion".
    stop_reason: str = "rounds_complete"


# ---------------------------------------------------------------------------
# Campaign configuration
# ---------------------------------------------------------------------------


SEARCH_MODES = ("full", "no-grid", "random-cell", "seed-only")


@dataclass
class CampaignConfig:
    """Configuration for a campaign.

    ``round_budget`` is the number of conversations **per round**, split
    across three buckets:
    - seed: replay seed corpus scenarios as an unsteered baseline
    - coverage: generate new scenarios for under-covered cells
    - mutation: mutate elite scenarios within their cells

    The total campaign budget is ``round_budget × num_rounds`` (set on
    ``run_campaign``).  ``conversations_per_round`` passed to
    ``run_campaign`` overrides ``round_budget`` for the run.
    """

    round_budget: int = 60

    # --- Three-bucket allocation (time-annealed schedule) ---
    # Each campaign round, ``rebalance(round_num, total_rounds)`` interpolates
    # the current ``alpha_*`` from these endpoints:
    #   • seed     — constant 0.15 stratified baseline (no-steering control)
    #   • coverage — directed scenario synthesis; anneals down 0.40 → 0.25
    #   • mutation — within-cell mutations of elites; anneals up 0.45 → 0.60.
    # The schedule is open-loop (AFLGo-style temperature schedule), so the
    # campaign trajectory is deterministic given (n_rounds, endpoints).
    alpha_seed_start: float = 0.15
    alpha_seed_end: float = 0.15
    alpha_coverage_start: float = 0.40
    alpha_coverage_end: float = 0.25
    alpha_mutation_start: float = 0.45
    alpha_mutation_end: float = 0.60
    alpha_min: float = 0.10
    """Floor on each bucket's weight; renormalization preserves sum-to-one."""

    enable_annealing: bool = True
    """Disable for constant-α campaigns."""

    # --- Current α values (state, mutated by rebalance()) ---
    # Initialised at the schedule's ``_start`` endpoints.
    alpha_seed: float = 0.15
    alpha_coverage: float = 0.40
    alpha_mutation: float = 0.45

    # --- Other knobs ---
    coverage_threshold: float = 0.8
    num_turns: int = 8
    max_mutation_depth: int = 2
    """Maximum lineage depth for an elite to remain a mutation parent.

    Chains of length > ``max_mutation_depth`` (i.e., depth-N+ scenarios)
    are excluded from the mutation bucket's elite pool.  At
    ``max_mutation_depth = 2`` the lineage tops out at three nodes
    (original → child → grandchild) — deep enough to compose two
    different within-cell perturbations on a discovered elite, shallow
    enough that a violation that persists under one perturbation
    still attaches to a discovered scenario rather than a
    synthetic chain.  See ``planner.mutation.mutation_depth``."""
    max_concurrent: int | None = None
    """Max concurrent conversations per round (passed as ``max_samples``
    to ``inspect_eval``).  ``None`` means Inspect's default (sequential)."""
    log_dir: str | None = None
    """Where Inspect writes its evaluation logs (Inspect's default when None)."""

    # Exhaustion backoff. On by default: monitor exhaustion can concentrate
    # in a few cells, and their priority should decay so the schedule moves
    # on. Counters reset at campaign start, not at round boundary.
    use_exhaustion_backoff: bool = True

    # Breadth-first gating threshold for prioritized_cells. While any
    # reachable cell has fewer than ``n_min_breadth`` observations, the
    # violation-weighting term is zeroed so budget flows to under-covered
    # cells. Set to 0 to disable.
    n_min_breadth: int = 3

    # Safety stop: abort the campaign after a round if the monitor-
    # exhaustion rate (termination because the role gate or the validate
    # step rejected every revision attempt) exceeds this fraction. Set to
    # None to disable.
    # A high exhaustion rate signals either a prompt/role regression
    # or a construct-validity failure; in either case the next round's
    # mutations will inherit the bad signal, so stopping is cheaper than
    # continuing. The threshold is checked against the just-completed
    # round only, not the campaign cumulative rate.
    monitor_exhaustion_abort_threshold: float | None = None

    # --- Search modes (see ``for_search``) ---
    # cell_selection controls how the coverage bucket picks cells:
    #   "coverage" — under-covered-first (the default full system)
    #   "random"   — uniform random cells (isolates coverage-directed
    #                targeting from targeting in general). Seed and mutation
    #                buckets are unchanged, so only the targeting policy moves.
    cell_selection: str = "coverage"
    # When True, bypass the grid entirely: every round generates
    # ``round_budget`` scenarios via no_grid (prompt-only) generation across
    # the role's principles — no seed/coverage/mutation split, no cell
    # targeting, no diagnostics. Isolates the whole grid apparatus from simply
    # asking a capable LLM for targeted pressure. The grid is still updated
    # post-hoc from the classifier so incidental coverage can be reported.
    no_grid: bool = False

    # Manual mid-campaign stop. Pass num_rounds=N (which sets the
    # annealing schedule) but exit cleanly after round stop_after_round.
    # The campaign can later be resumed from round stop_after_round+1
    # with the same num_rounds, preserving the original alpha schedule.
    # Set to None (default) to run all rounds.
    stop_after_round: int | None = None

    # Web-agent execution binds generated scenarios to concrete benchmark
    # tasks. When enabled, the seed supplying that immutable execution frame
    # must target the same principle as the requested coverage cell.
    # Conversational campaigns sample frames without this restriction.
    require_principle_matched_execution_anchor: bool = False

    @classmethod
    def for_search(cls, search: str, **settings) -> "CampaignConfig":
        """Configuration for one of ``SEARCH_MODES``.

        ``full`` is the default system; ``random-cell`` picks coverage cells
        uniformly at random; ``no-grid`` generates every scenario from the
        role spec alone; ``seed-only`` spends the whole budget on seeds.
        """
        if search not in SEARCH_MODES:
            raise ValueError(f"Unknown search mode {search!r}; expected one of {SEARCH_MODES}")
        if search == "seed-only":
            settings.update(
                enable_annealing=False,
                alpha_seed_start=1.0, alpha_seed_end=1.0,
                alpha_coverage_start=0.0, alpha_coverage_end=0.0,
                alpha_mutation_start=0.0, alpha_mutation_end=0.0,
            )
        return cls(
            cell_selection="random" if search == "random-cell" else "coverage",
            no_grid=search == "no-grid",
            **settings,
        )

    def __post_init__(self) -> None:
        # Initialise current α to the start endpoints (round 1 weights).
        self.alpha_seed = self.alpha_seed_start
        self.alpha_coverage = self.alpha_coverage_start
        self.alpha_mutation = self.alpha_mutation_start
        # Validate endpoints sum to 1 at both ends of the schedule.
        for label, s in (
            ("start", self.alpha_seed_start + self.alpha_coverage_start + self.alpha_mutation_start),
            ("end",   self.alpha_seed_end   + self.alpha_coverage_end   + self.alpha_mutation_end),
        ):
            if abs(s - 1.0) > 0.01:
                raise ValueError(
                    f"Schedule {label} endpoints must sum to 1.0, got {s:.3f}"
                )

    def bucket_sizes(self, round_budget: int | None = None) -> dict[str, int]:
        """Split a round budget across the three buckets (seed/coverage/mutation).

        Mutation absorbs rounding so total == round_budget.
        """
        budget = round_budget or self.round_budget
        seed = int(budget * self.alpha_seed)
        coverage = int(budget * self.alpha_coverage)
        mutation = budget - seed - coverage
        return {
            "seed": seed,
            "coverage": coverage,
            "mutation": mutation,
        }

    def rebalance(
        self,
        round_num: int,
        total_rounds: int,
    ) -> None:
        """Time-annealed bucket schedule (AFLGo-style open-loop temperature).

        Linearly interpolates each bucket's α from its ``_start`` to ``_end``
        endpoint as ``t = (round_num - 1) / (total_rounds - 1)``. After the
        schedule applies, floors at ``alpha_min`` are enforced and the
        weights are renormalised so they still sum to 1. The schedule is
        open-loop: a single noisy round cannot shrink a bucket for the rest
        of the campaign.
        """
        if not self.enable_annealing:
            return

        if total_rounds <= 1:
            t = 0.0
        else:
            t = max(0.0, min(1.0, (round_num - 1) / (total_rounds - 1)))

        seed = (1 - t) * self.alpha_seed_start + t * self.alpha_seed_end
        cov  = (1 - t) * self.alpha_coverage_start + t * self.alpha_coverage_end
        mut  = (1 - t) * self.alpha_mutation_start + t * self.alpha_mutation_end

        # Apply floors
        seed = max(self.alpha_min, seed)
        cov  = max(self.alpha_min, cov)
        mut  = max(self.alpha_min, mut)

        # Renormalise so floors don't break sum-to-1
        total = seed + cov + mut
        self.alpha_seed = seed / total
        self.alpha_coverage = cov / total
        self.alpha_mutation = mut / total

        logger.info(
            "Annealed alphas (round %d/%d, t=%.2f): seed=%.2f cov=%.2f mut=%.2f",
            round_num, total_rounds, t,
            self.alpha_seed, self.alpha_coverage, self.alpha_mutation,
        )
