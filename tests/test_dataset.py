"""Tests for dataset construction."""

from rolefuzzer.dataset import role_dataset_from_scenarios
from rolefuzzer.spec import Principle, RoleSpec, Scenario, ViolationCriteria


class TestTargetingMetadata:
    """Targeting fields on a Scenario reach Sample.metadata, where the
    monitor reads the per-sample target cell."""

    def _role_spec(self) -> RoleSpec:
        vc = ViolationCriteria(
            clear_breach="b", partial_deviation="p", full_adherence="a",
        )
        return RoleSpec(
            id="tutor", name="Tutor", description="d",
            system_prompt="You are a tutor. {principles_block}",
            principles=[Principle(id="p1", name="P1", specification="spec", violation_criteria=vc)],
        )

    def test_target_fields_on_metadata(self):
        scenario = Scenario(
            id="cell_targeted",
            role_id="tutor",
            target_principle_id="p1",
            name="Targeted",
            description="d",
            user_scenario="us",
            user_principle="up",
            target_mode_id="3",
            target_strategy_id="5.2",
            parent_scenario_id="seed_1",
            mutation_type="intensity",
        )
        ds = role_dataset_from_scenarios(self._role_spec(), [scenario])
        md = ds[0].metadata
        assert md["target_mode_id"] == "3"
        assert md["target_strategy_id"] == "5.2"
        assert md["parent_scenario_id"] == "seed_1"
        assert md["mutation_type"] == "intensity"

    def test_untargeted_scenario_has_no_target_fields(self):
        scenario = Scenario(
            id="untargeted",
            role_id="tutor",
            target_principle_id="p1",
            name="Untargeted",
            description="d",
            user_scenario="us",
            user_principle="up",
        )
        ds = role_dataset_from_scenarios(self._role_spec(), [scenario])
        md = ds[0].metadata
        assert md["target_mode_id"] is None
        assert md["target_strategy_id"] is None
        assert md["parent_scenario_id"] is None
        assert md["mutation_type"] is None
