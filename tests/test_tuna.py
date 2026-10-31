"""Tests for TUNA taxonomy types, loading, and rendering."""

import pytest

from rolefuzzer.taxonomy import (
    TunaMode,
    TunaRequestType,
    TunaStrategy,
    TunaTaxonomy,
    load_taxonomy,
)
from tests.conftest import REPO_ROOT

TAXONOMY_PATH = REPO_ROOT / "taxonomy" / "tuna.yaml"


class TestTunaRequestType:
    def test_frozen(self):
        rt = TunaRequestType(id="1.1.1", name="Test", description="A test type")
        with pytest.raises(Exception):
            rt.id = "changed"

    def test_examples_default(self):
        rt = TunaRequestType(id="1.1.1", name="Test", description="A test type")
        assert rt.examples == []

    def test_examples_provided(self):
        rt = TunaRequestType(
            id="1.1.1", name="Test", description="A test type",
            examples=["example 1", "example 2"],
        )
        assert len(rt.examples) == 2


class TestTunaTaxonomy:
    @pytest.fixture
    def mini_taxonomy(self) -> TunaTaxonomy:
        return TunaTaxonomy(
            modes=[
                TunaMode(
                    id="1",
                    name="Info Seeking",
                    strategies=[
                        TunaStrategy(
                            id="1.1",
                            name="Retrieval",
                            request_types=[
                                TunaRequestType(id="1.1.1", name="Fact Q", description="fact"),
                                TunaRequestType(id="1.1.2", name="Def Req", description="def"),
                            ],
                        ),
                    ],
                ),
                TunaMode(
                    id="2",
                    name="Content Gen",
                    strategies=[
                        TunaStrategy(
                            id="2.1",
                            name="Composition",
                            request_types=[
                                TunaRequestType(id="2.1.1", name="Essay", description="essay"),
                            ],
                        ),
                    ],
                ),
            ]
        )

    def test_frozen(self, mini_taxonomy):
        with pytest.raises(Exception):
            mini_taxonomy.modes = []


class TestTunaTaxonomyNavigation:
    """Tests for navigation helpers on TunaTaxonomy."""

    @pytest.fixture
    def nav_taxonomy(self) -> TunaTaxonomy:
        """Taxonomy with modes 1-6 (2 instrumental, 2 relational-ish) for testing."""
        return TunaTaxonomy(
            modes=[
                TunaMode(
                    id="1",
                    name="Info Seeking",
                    strategies=[
                        TunaStrategy(
                            id="1.1",
                            name="Retrieval",
                            request_types=[
                                TunaRequestType(id="1.1.1", name="Fact Q", description="fact"),
                            ],
                        ),
                    ],
                ),
                TunaMode(
                    id="2",
                    name="Content Gen",
                    strategies=[
                        TunaStrategy(
                            id="2.1",
                            name="Composition",
                            request_types=[
                                TunaRequestType(id="2.1.1", name="Essay", description="essay"),
                            ],
                        ),
                    ],
                ),
                TunaMode(
                    id="3",
                    name="Procedural Guidance",
                    strategies=[
                        TunaStrategy(
                            id="3.1",
                            name="Instruction",
                            request_types=[
                                TunaRequestType(id="3.1.1", name="How-To", description="how"),
                            ],
                        ),
                    ],
                ),
                TunaMode(
                    id="4",
                    name="Content Transformation",
                    strategies=[
                        TunaStrategy(
                            id="4.1",
                            name="Translation",
                            request_types=[
                                TunaRequestType(id="4.1.1", name="Translate", description="xlate"),
                            ],
                        ),
                    ],
                ),
                TunaMode(
                    id="5",
                    name="Social Interaction",
                    strategies=[
                        TunaStrategy(
                            id="5.1",
                            name="Sociability",
                            request_types=[
                                TunaRequestType(id="5.1.1", name="Greeting", description="hi"),
                            ],
                        ),
                        TunaStrategy(
                            id="5.2",
                            name="Shared Understanding",
                            request_types=[
                                TunaRequestType(id="5.2.1", name="Clarify", description="clarify"),
                            ],
                        ),
                    ],
                ),
                TunaMode(
                    id="6",
                    name="Meta-Conversation",
                    strategies=[
                        TunaStrategy(
                            id="6.1",
                            name="System Management",
                            request_types=[
                                TunaRequestType(id="6.1.1", name="Reset", description="reset"),
                            ],
                        ),
                    ],
                ),
            ]
        )

    def test_get_mode(self, nav_taxonomy):
        mode = nav_taxonomy.get_mode("1")
        assert mode.name == "Info Seeking"

    def test_get_mode_not_found(self, nav_taxonomy):
        with pytest.raises(ValueError, match="not found"):
            nav_taxonomy.get_mode("99")

    def test_get_strategy(self, nav_taxonomy):
        strategy = nav_taxonomy.get_strategy("5.2")
        assert strategy.name == "Shared Understanding"

    def test_get_strategy_not_found(self, nav_taxonomy):
        with pytest.raises(ValueError, match="not found"):
            nav_taxonomy.get_strategy("9.9")

    def test_instrumental_modes(self, nav_taxonomy):
        modes = nav_taxonomy.instrumental_modes()
        assert len(modes) == 4
        assert {m.id for m in modes} == {"1", "2", "3", "4"}

    def test_relational_strategies(self, nav_taxonomy):
        strategies = nav_taxonomy.relational_strategies()
        assert len(strategies) == 3
        assert {s.id for s in strategies} == {"5.1", "5.2", "6.1"}


class TestLoadTaxonomy:
    def test_load_full_taxonomy(self):
        taxonomy = load_taxonomy(TAXONOMY_PATH)
        # 6 modes
        assert len(taxonomy.modes) == 6

        # Mode names as defined by TUNA (arXiv:2510.06124)
        mode_names = [m.name for m in taxonomy.modes]
        assert "Information Seeking" in mode_names
        assert "Information Processing & Synthesis" in mode_names
        assert "Procedural Guidance & Execution" in mode_names
        assert "Content Creation & Transformation" in mode_names
        assert "Social Interaction" in mode_names
        assert "Meta-Conversation" in mode_names

    def test_57_request_types(self):
        taxonomy = load_taxonomy(TAXONOMY_PATH)
        ids = [rt.id for m in taxonomy.modes for s in m.strategies for rt in s.request_types]
        assert len(ids) == 57

    def test_14_strategies(self):
        taxonomy = load_taxonomy(TAXONOMY_PATH)
        strategies = []
        for mode in taxonomy.modes:
            strategies.extend(mode.strategies)
        assert len(strategies) == 14

    def test_examples_loaded(self):
        taxonomy = load_taxonomy(TAXONOMY_PATH)
        rt = taxonomy.modes[0].strategies[0].request_types[0]
        assert rt.id == "1.1.1"
        assert len(rt.examples) > 0
        assert "tokyo" in rt.examples[0].lower()
