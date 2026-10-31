"""TUNA taxonomy types and utilities.

Frozen Pydantic models for the TUNA taxonomy (Taxonomy of User-LLM
Natural-language Actions): 6 modes, 14 strategies, 57 request types.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Taxonomy structure models
# ---------------------------------------------------------------------------


class TunaRequestType(BaseModel, frozen=True):
    """A single request type in the TUNA taxonomy (leaf node)."""

    id: str
    name: str
    description: str
    examples: list[str] = Field(default_factory=list)


class TunaStrategy(BaseModel, frozen=True):
    """A strategy grouping related request types."""

    id: str
    name: str
    request_types: list[TunaRequestType]


class TunaMode(BaseModel, frozen=True):
    """A top-level mode grouping strategies."""

    id: str
    name: str
    strategies: list[TunaStrategy]


class TunaTaxonomy(BaseModel, frozen=True):
    """Full TUNA taxonomy (6 modes, 14 strategies, 57 request types)."""

    modes: list[TunaMode]

    def render_modes_summary(self) -> str:
        """Render a compact mode-and-strategy summary (no request types).

        Suitable for prompts where the full taxonomy would be too large but
        the planner still needs the vocabulary of interaction modes to
        describe realistic behavioral shifts.
        """
        lines: list[str] = []
        for mode in self.modes:
            strategies = ", ".join(s.name for s in mode.strategies)
            lines.append(f"Mode {mode.id} — {mode.name}: {strategies}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # navigation helpers
    # ------------------------------------------------------------------

    def get_mode(self, mode_id: str) -> TunaMode:
        """Look up a mode by ID (e.g. '1'). Raises ValueError if not found."""
        for mode in self.modes:
            if mode.id == mode_id:
                return mode
        available = [m.id for m in self.modes]
        raise ValueError(
            f"Mode '{mode_id}' not found in taxonomy. Available: {available}"
        )

    def get_strategy(self, strategy_id: str) -> TunaStrategy:
        """Look up a strategy by ID (e.g. '1.1'). Raises ValueError if not found."""
        for mode in self.modes:
            for strategy in mode.strategies:
                if strategy.id == strategy_id:
                    return strategy
        available = [s.id for m in self.modes for s in m.strategies]
        raise ValueError(
            f"Strategy '{strategy_id}' not found in taxonomy. Available: {available}"
        )

    def instrumental_modes(self) -> list[TunaMode]:
        """Return modes 1-4 (instrumental: what the user does)."""
        return [m for m in self.modes if m.id in ("1", "2", "3", "4")]

    def relational_strategies(self) -> list[TunaStrategy]:
        """Return strategies from modes 5-6 (relational/meta: how the user does it)."""
        return [
            s
            for m in self.modes
            if m.id in ("5", "6")
            for s in m.strategies
        ]

    def targetable_relational_strategies(self) -> list[TunaStrategy]:
        """Return relational strategies suitable for the coverage grid.

        Excludes strategy 6.3 (Communicative Status) which represents
        meta-communication behaviors that cannot be meaningfully targeted
        by the planner — they emerge organically from conversation dynamics.
        """
        return [s for s in self.relational_strategies() if s.id != "6.3"]


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def load_taxonomy(path: str | Path) -> TunaTaxonomy:
    """Load a TunaTaxonomy from a YAML file.

    Expected YAML structure::

        taxonomy:
          modes:
            - id: "1"
              name: "Information Seeking"
              strategies:
                - id: "1.1"
                  name: "Retrieval"
                  request_types:
                    - id: "1.1.1"
                      name: "Direct Fact Question"
                      description: "..."
                      examples: [...]
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return TunaTaxonomy(**raw["taxonomy"])
