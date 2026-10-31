"""Tests for domain types."""


import pytest

from rolefuzzer.spec import (
    MutationType,
    Scenario,
    ViolationStatus,
    load_role,
    render_violation_criteria,
)
from tests.conftest import ROLES_DIR


class TestViolationCriteria:
    def test_frozen(self, sample_violation_criteria):
        with pytest.raises(Exception):
            sample_violation_criteria.clear_breach = "changed"


class TestPrinciple:
    def test_frozen(self, sample_principle):
        with pytest.raises(Exception):
            sample_principle.id = "changed"


class TestRoleSpec:
    def test_render_system_prompt(self, sample_role):
        rendered = sample_role.render_system_prompt()
        assert "STUDENTS MUST DO THE WORK" in rendered
        assert "Guide students" in rendered
        assert "{principles_block}" not in rendered

    def test_get_principle(self, sample_role):
        p = sample_role.get_principle("academic_integrity")
        assert p.id == "academic_integrity"

    def test_get_principle_not_found(self, sample_role):
        with pytest.raises(ValueError, match="not found"):
            sample_role.get_principle("nonexistent")

    def test_frozen(self, sample_role):
        with pytest.raises(Exception):
            sample_role.id = "changed"


class TestScenario:
    def test_frozen(self, sample_scenario):
        with pytest.raises(Exception):
            sample_scenario.id = "changed"


class TestMutationType:
    def test_values(self):
        assert MutationType.INTENSITY == "intensity"
        assert MutationType.FRAME_SWAP == "frame_swap"

    def test_only_two_operators(self):
        assert set(MutationType) == {MutationType.INTENSITY, MutationType.FRAME_SWAP}

    def test_is_string_enum(self):
        assert isinstance(MutationType.INTENSITY, str)


class TestScenarioTargetFields:
    """Tests for optional fields on Scenario."""

    def test_target_fields_default_none(self):
        s = Scenario(
            id="test",
            role_id="tutor",
            target_principle_id="p1",
            name="Test",
            description="Test",
            user_scenario="scenario",
            user_principle="principle",
        )
        assert s.target_mode_id is None
        assert s.target_strategy_id is None
        assert s.parent_scenario_id is None
        assert s.mutation_type is None


class TestViolationStatus:
    def test_values(self):
        assert ViolationStatus.FULL_ADHERENCE == "full_adherence"
        assert ViolationStatus.PARTIAL_DEVIATION == "partial_deviation"
        assert ViolationStatus.CLEAR_BREACH == "clear_breach"


class TestRenderViolationCriteria:
    def test_render(self, sample_principle):
        text = render_violation_criteria(sample_principle)
        assert "Principle: Students Must Do the Work" in text
        assert "clear_breach" in text
        assert "partial_deviation" in text
        assert "full_adherence" in text


class TestYAMLLoading:
    def test_load_tutor_role(self):
        role_spec = load_role(ROLES_DIR / "tutor.yaml")
        assert role_spec.id == "tutor"
        assert len(role_spec.principles) == 4
        assert role_spec.get_principle("academic_integrity")
        assert role_spec.get_principle("confidentiality")
        assert role_spec.get_principle("bounded_pedagogical_role")
        assert role_spec.get_principle("individual_differences")
        rendered = role_spec.render_system_prompt()
        assert "{principles_block}" not in rendered

# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------


class TestRoleModalityContract:
    @pytest.mark.parametrize(
        "role_id", ["tutor", "financial_advisor", "interviewer", "antagonist"]
    )
    def test_role_declares_user_role_description(self, role_id):
        role_spec = load_role(ROLES_DIR / f"{role_id}.yaml")
        assert role_spec.user_role_description.strip(), (
            f"role {role_id} must declare user_role_description"
        )
        assert "in_role_interaction" in role_spec.allowed_interaction_modalities

    def test_scenario_defaults_to_in_role_modality(self):
        from rolefuzzer.spec import DEFAULT_INTERACTION_MODALITY

        s = Scenario(
            id="x",
            role_id="tutor",
            target_principle_id="academic_integrity",
            name="n",
            description="d",
            user_scenario="u",
            user_principle="g",
        )
        assert s.user_role == ""
        assert s.interaction_modality == DEFAULT_INTERACTION_MODALITY

    def test_ensure_scenario_role_fields_backfills(self):
        from rolefuzzer.spec import (
            DEFAULT_INTERACTION_MODALITY,
            ensure_scenario_role_fields,
        )

        role_spec = load_role(ROLES_DIR / "tutor.yaml")
        s = Scenario(
            id="x",
            role_id="tutor",
            target_principle_id="academic_integrity",
            name="n",
            description="d",
            user_scenario="u",
            user_principle="g",
            user_role="",
            interaction_modality="",
        )
        s2 = ensure_scenario_role_fields(s, role_spec)
        assert s2.user_role == role_spec.user_role_description.strip()
        assert s2.interaction_modality == DEFAULT_INTERACTION_MODALITY

    def test_ensure_scenario_role_fields_preserves_set_values(self):
        from rolefuzzer.spec import ensure_scenario_role_fields

        role_spec = load_role(ROLES_DIR / "tutor.yaml")
        s = Scenario(
            id="x",
            role_id="tutor",
            target_principle_id="academic_integrity",
            name="n",
            description="d",
            user_scenario="u",
            user_principle="g",
            user_role="overcaffeinated freshman",
            interaction_modality="in_role_interaction",
        )
        s2 = ensure_scenario_role_fields(s, role_spec)
        assert s2.user_role == "overcaffeinated freshman"
        assert s2.interaction_modality == "in_role_interaction"
