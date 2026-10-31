"""Campaign planner: coverage grid, scenario generation, mutation and the campaign loop."""

from rolefuzzer.planner.config import (
    CampaignConfig,
    RoundReport,
)
from rolefuzzer.planner.diagnostics import (
    render_diagnostic_context,
)
from rolefuzzer.planner.generation import (
    generate_scenario_for_cell,
)
from rolefuzzer.planner.grid import (
    CoverageGrid,
    CoverageGridCell,
    FeasibilityTable,
    QualityFeasibilityTable,
    ScenarioOutcomeTracker,
)
from rolefuzzer.planner.loop import (
    chat_rollout,
    run_campaign,
)
from rolefuzzer.planner.mutation import (
    mutate_scenario,
    mutation_depth,
)
from rolefuzzer.planner.scenarios import (
    PlannerOutputError,
)

__all__ = [
    "CampaignConfig",
    "CoverageGrid",
    "CoverageGridCell",
    "FeasibilityTable",
    "PlannerOutputError",
    "QualityFeasibilityTable",
    "RoundReport",
    "ScenarioOutcomeTracker",
    "chat_rollout",
    "generate_scenario_for_cell",
    "mutate_scenario",
    "mutation_depth",
    "render_diagnostic_context",
    "run_campaign",
]
