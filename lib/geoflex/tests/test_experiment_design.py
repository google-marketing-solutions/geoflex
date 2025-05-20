"""Tests for the experiment design module."""

import geoflex.experiment_design
import geoflex.metrics
import numpy as np
import pandas as pd
import pytest


GeoAssignment = geoflex.experiment_design.GeoAssignment
ExperimentDesign = geoflex.experiment_design.ExperimentDesign
GeoEligibility = geoflex.experiment_design.GeoEligibility
ExperimentBudget = geoflex.experiment_design.ExperimentBudget
ExperimentBudgetType = geoflex.experiment_design.ExperimentBudgetType
CellVolumeConstraint = geoflex.experiment_design.CellVolumeConstraint
CellVolumeConstraintType = geoflex.experiment_design.CellVolumeConstraintType

# Tests don't need docstrings.
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name


@pytest.mark.parametrize(
    "treatment,control,exclude",
    [
        ([["US", "UK"], []], ["US", "AU"], ["FR"]),  # Treatment control overlap
        (
            [["US", "UK"], ["CA", "US"]],
            ["FR", "AU"],
            [],
        ),  # Treatment treatment overlap
        ([["US", "UK"], []], ["FR", "AU"], ["US"]),  # Treatment exclude overlap
        ([["US", "UK"], []], ["FR", "AU"], ["AU"]),  # Control exclude overlap
    ],
)
def test_geo_assignment_raises_exception_if_geos_in_multiple_groups(
    treatment, control, exclude
):
  treatment_sets = [set(arm) for arm in treatment]
  control_set = set(control)
  exclude_set = set(exclude)
  with pytest.raises(ValueError, match="multiple groups"):
    GeoAssignment(
        treatment=treatment_sets, control=control_set, exclude=exclude_set
    )


def test_geo_assignment_raises_exception_if_treatment_geos_are_single_list():
  with pytest.raises(ValueError):
    # Treatment must be a list of lists to be able to have multiple treatment
    # arms.
    GeoAssignment(treatment=["US", "UK"], control=["US", "AU"])


def test_metric_names_must_be_unique():
  with pytest.raises(ValueError):
    ExperimentDesign(
        primary_metric="revenue",
        experiment_budget=ExperimentBudget(
            value=-0.1,
            budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
        ),
        secondary_metrics=["revenue", "conversions"],
        methodology="test_methodology",
        runtime_weeks=4,
        alpha=0.1,
        geo_eligibility=None,
    )


@pytest.mark.parametrize(
    "cost_metric", [geoflex.metrics.iROAS(), geoflex.metrics.CPiA()]
)
@pytest.mark.parametrize("budget_values", [0.0, [-1.0, 0.0]])
def test_budget_must_be_non_zero_if_cost_metrics_are_used(
    budget_values, cost_metric
):
  with pytest.raises(ValueError):
    ExperimentDesign(
        primary_metric="revenue",
        experiment_budget=ExperimentBudget(
            value=budget_values,
            budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
        ),
        secondary_metrics=["conversions", cost_metric],
        methodology="test_methodology",
        runtime_weeks=4,
        alpha=0.1,
        geo_eligibility=None,
        n_cells=3,
    )


def test_budget_must_be_consistent_with_n_cells():
  with pytest.raises(ValueError):
    ExperimentDesign(
        primary_metric="revenue",
        experiment_budget=ExperimentBudget(
            value=[-0.1, -0.2, -0.3],
            budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
        ),
        secondary_metrics=["conversions", geoflex.metrics.iROAS()],
        methodology="test_methodology",
        runtime_weeks=4,
        alpha=0.1,
        n_cells=3,
        geo_eligibility=None,
    )


def test_cost_cell_volume_constraint_requires_cost_metrics():
  with pytest.raises(ValueError):
    ExperimentDesign(
        primary_metric="revenue",
        experiment_budget=ExperimentBudget(
            value=-0.1,
            budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
        ),
        secondary_metrics=["conversions"],
        methodology="test_methodology",
        runtime_weeks=4,
        alpha=0.1,
        n_cells=3,
        geo_eligibility=None,
        cell_volume_constraint=CellVolumeConstraint(
            values=[0.1, 0.2, 0.3],
            constraint_type=CellVolumeConstraintType.MAX_PERCENTAGE_OF_TOTAL_COST,
        ),
    )


def test_cost_cell_volume_constraint_works_with_cost_metrics():
  # Should not raise an error
  ExperimentDesign(
      primary_metric="revenue",
      experiment_budget=ExperimentBudget(
          value=-0.1,
          budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
      ),
      secondary_metrics=["conversions", geoflex.metrics.iROAS()],
      methodology="test_methodology",
      runtime_weeks=4,
      alpha=0.1,
      n_cells=3,
      geo_eligibility=None,
      cell_volume_constraint=CellVolumeConstraint(
          values=[0.1, 0.2, 0.3],
          constraint_type=CellVolumeConstraintType.MAX_PERCENTAGE_OF_TOTAL_COST,
      ),
  )


@pytest.mark.parametrize(
    "constraint_type",
    [
        CellVolumeConstraintType.MAX_PERCENTAGE_OF_TOTAL_RESPONSE,
        CellVolumeConstraintType.MAX_GEOS,
    ],
)
def test_non_cost_cell_volume_constraint_does_not_require_cost_metrics(
    constraint_type,
):
  # Should not raise an error
  ExperimentDesign(
      primary_metric="revenue",
      experiment_budget=ExperimentBudget(
          value=-0.1,
          budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
      ),
      secondary_metrics=["conversions"],
      methodology="test_methodology",
      runtime_weeks=4,
      alpha=0.1,
      n_cells=3,
      geo_eligibility=None,
      cell_volume_constraint=CellVolumeConstraint(
          values=[0.1, 0.2, 0.3], constraint_type=constraint_type
      ),
  )


@pytest.mark.parametrize(
    "primary_metric,secondary_metrics,expected_column",
    [
        (geoflex.metrics.iROAS(cost_column="cost_1"), ["revenue"], "cost_1"),
        (geoflex.metrics.CPiA(cost_column="cost_2"), ["revenue"], "cost_2"),
        (
            geoflex.metrics.iROAS(cost_column="cost_3"),
            [geoflex.metrics.CPiA(cost_column="cost_4")],
            "cost_3",
        ),
        ("revenue", [geoflex.metrics.iROAS(cost_column="cost_5")], "cost_5"),
        (
            "revenue",
            [
                geoflex.metrics.CPiA(cost_column="cost_6"),
                geoflex.metrics.iROAS(cost_column="cost_7"),
            ],
            "cost_6",
        ),
        ("revenue", [], None),
        ("revenue", ["conversions"], None),
    ],
)
def test_main_cost_column_returns_expected_column(
    primary_metric, secondary_metrics, expected_column
):
  design = ExperimentDesign(
      primary_metric=primary_metric,
      secondary_metrics=secondary_metrics,
      methodology="test_methodology",
      runtime_weeks=4,
      experiment_budget=ExperimentBudget(
          value=-0.1,
          budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
      ),
  )
  assert design.main_cost_column == expected_column


def test_geoeligibilty_inflexible_2cell():
  eligibility = GeoEligibility(
      control={"g1", "g2"},
      treatment=[{"g3"}],  # 1 treatment arm, 2 cells total
      exclude={"g4"},
      all_geos={"g1", "g2", "g3", "g4", "g5"},
      flexible=False,
  )
  data_frames = eligibility.data

  # check there is only 1 dataframe
  assert isinstance(data_frames, list)
  assert len(data_frames) == 1
  df = data_frames[0]

  assert isinstance(df, pd.DataFrame)
  assert df.index.name == "geo"
  assert df.columns.to_list() == ["control", "treatment", "exclude"]
  assert set(df.index) == {"g1", "g2", "g3", "g4", "g5"}

  # check values of specific geos
  assert df.loc["g1"].equals(
      pd.Series({"control": 1, "treatment": 0, "exclude": 0}, name="g1")
  )
  assert df.loc["g2"].equals(
      pd.Series({"control": 1, "treatment": 0, "exclude": 0}, name="g2")
  )
  assert df.loc["g3"].equals(
      pd.Series({"control": 0, "treatment": 1, "exclude": 0}, name="g3")
  )
  assert df.loc["g4"].equals(
      pd.Series({"control": 0, "treatment": 0, "exclude": 1}, name="g4")
  )
  # g5 is not explicitly defined and flexible=False -> should be all 0s
  assert df.loc["g5"].equals(
      pd.Series({"control": 0, "treatment": 0, "exclude": 0}, name="g5")
  )


def test_geoeligibility_flexible_2cell():
  eligibility = GeoEligibility(
      control={"g4", "g3"},  # out of order numbers
      treatment=[{"g2"}],
      exclude={"g5"},
      all_geos={"g1", "g2", "g3", "g4", "g5", "g6"},
      flexible=True,
  )
  data_frames = eligibility.data

  # check there is only 1 dataframe
  assert isinstance(data_frames, list)
  assert len(data_frames) == 1
  df = data_frames[0]

  assert isinstance(df, pd.DataFrame)
  assert df.index.name == "geo"
  assert df.columns.to_list() == ["control", "treatment", "exclude"]
  assert set(df.index) == {"g1", "g2", "g3", "g4", "g5", "g6"}

  # check values of specific geos
  # g1 is not defined and flexible=True -> flexible control and treatment
  assert df.loc["g1"].equals(
      pd.Series({"control": 1, "treatment": 1, "exclude": 0}, name="g1")
  )
  assert df.loc["g2"].equals(
      pd.Series({"control": 0, "treatment": 1, "exclude": 0}, name="g2")
  )
  assert df.loc["g3"].equals(
      pd.Series({"control": 1, "treatment": 0, "exclude": 0}, name="g3")
  )
  assert df.loc["g4"].equals(
      pd.Series({"control": 1, "treatment": 0, "exclude": 0}, name="g4")
  )
  assert df.loc["g5"].equals(
      pd.Series({"control": 0, "treatment": 0, "exclude": 1}, name="g5")
  )
  # g6 is not defined and flexible=True -> flexible control and treatment
  assert df.loc["g6"].equals(
      pd.Series({"control": 1, "treatment": 1, "exclude": 0}, name="g6")
  )


def test_geoeligibility_multiarm_3cell():
  eligibililty = GeoEligibility(
      control={"c1"},
      treatment=[{"t1", "t3"}, {"t2", "t3"}],
      exclude={"e1"},
      all_geos={"c1", "t1", "t2", "t3", "e1", "u1"},
      flexible=False,
  )
  data_frames = eligibililty.data

  # check there are 2 dataframes
  assert isinstance(data_frames, list)
  assert len(data_frames) == 2
  df_arm1 = data_frames[0]
  df_arm2 = data_frames[1]

  assert isinstance(df_arm1, pd.DataFrame)
  assert isinstance(df_arm2, pd.DataFrame)
  expected_geos = {"c1", "t1", "t2", "t3", "e1", "u1"}
  assert set(df_arm1.index) == expected_geos
  assert set(df_arm2.index) == expected_geos

  # check eligibility for t1 which is only eligible for treatment in arm1
  assert df_arm1.loc["t1"].equals(
      pd.Series({"control": 0, "treatment": 1, "exclude": 0}, name="t1")
  )
  assert df_arm2.loc["t1"].equals(
      pd.Series({"control": 0, "treatment": 0, "exclude": 0}, name="t1")
  )

  # check eligibility for t2 which is only eligible for treatment in arm2
  assert df_arm1.loc["t2"].equals(
      pd.Series({"control": 0, "treatment": 0, "exclude": 0}, name="t2")
  )
  assert df_arm2.loc["t2"].equals(
      pd.Series({"control": 0, "treatment": 1, "exclude": 0}, name="t2")
  )

  # check eligibility for t3 which is eligible for treatment in both arms
  assert df_arm1.loc["t3"].equals(
      pd.Series({"control": 0, "treatment": 1, "exclude": 0}, name="t3")
  )
  assert df_arm2.loc["t3"].equals(
      pd.Series({"control": 0, "treatment": 1, "exclude": 0}, name="t3")
  )

  # check eligibility for c1 which is only eligible for control
  assert df_arm1.loc["c1"].equals(
      pd.Series({"control": 1, "treatment": 0, "exclude": 0}, name="c1")
  )
  assert df_arm2.loc["c1"].equals(
      pd.Series({"control": 1, "treatment": 0, "exclude": 0}, name="c1")
  )

  # check eligibility for e1 which is only eligible for exclude
  assert df_arm1.loc["e1"].equals(
      pd.Series({"control": 0, "treatment": 0, "exclude": 1}, name="e1")
  )
  assert df_arm2.loc["e1"].equals(
      pd.Series({"control": 0, "treatment": 0, "exclude": 1}, name="e1")
  )

  # check eligibility for u1 which is not defined and flexible=False
  assert df_arm1.loc["u1"].equals(
      pd.Series({"control": 0, "treatment": 0, "exclude": 0}, name="u1")
  )
  assert df_arm2.loc["u1"].equals(
      pd.Series({"control": 0, "treatment": 0, "exclude": 0}, name="u1")
  )


def test_make_geo_assignment_array_returns_correct_array():
  assignment = GeoAssignment(
      treatment=[{"US", "UK"}, {"CA"}],
      control={"FR", "AU"},
      exclude={"DE", "JP"},
  ).make_geo_assignment_array(["US", "CA", "UK", "FR", "DE", "Other"])
  expected_assignment = np.array([1, 2, 1, 0, -1, -1])
  assert np.array_equal(assignment, expected_assignment)


def test_can_write_design_to_json():
  design = ExperimentDesign(
      primary_metric="revenue",
      experiment_budget=ExperimentBudget(
          value=-0.1,
          budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
      ),
      secondary_metrics=["conversions"],
      methodology="test_methodology",
      runtime_weeks=4,
      n_cells=2,
      alpha=0.1,
      geo_eligibility=None,
  )
  design_as_json = design.model_dump_json()
  new_design = ExperimentDesign.model_validate_json(design_as_json)

  assert new_design == design


def test_can_write_design_to_dict():
  design = ExperimentDesign(
      primary_metric="revenue",
      experiment_budget=ExperimentBudget(
          value=-0.1,
          budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
      ),
      secondary_metrics=["conversions"],
      methodology="test_methodology",
      runtime_weeks=4,
      n_cells=2,
      alpha=0.1,
      geo_eligibility=None,
  )
  design_as_dict = design.model_dump()
  new_design = ExperimentDesign.model_validate(design_as_dict)

  assert new_design == design


@pytest.fixture(name="mock_design")
def mock_design_fixture():
  """Fixture for a mock design."""
  return ExperimentDesign(
      primary_metric="revenue",
      secondary_metrics=[
          geoflex.metrics.iROAS(),
          geoflex.metrics.CPiA(),
      ],
      experiment_budget=ExperimentBudget(
          value=-0.1,
          budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
      ),
      methodology="TestingMethodology",
      runtime_weeks=4,
      n_cells=3,
      alpha=0.1,
      geo_eligibility=None,
  )


@pytest.fixture(name="mock_design_evaluation_results")
def mock_design_evaluation_results_fixture():
  """Fixture for a mock design evaluation results."""
  return geoflex.evaluation.ExperimentDesignEvaluationResults(
      is_valid_design=True,
      primary_metric_name="revenue",
      alpha=0.1,
      alternative_hypothesis="two-sided",
      representiveness_scores_per_cell=[1.0, 2.0],
      all_metric_results_per_cell={
          "revenue": [
              geoflex.evaluation.SingleEvaluationResult(
                  standard_error_absolute_effect=1.3,
                  standard_error_relative_effect=1.2,
                  coverage_absolute_effect=0.4,
                  coverage_relative_effect=0.5,
                  all_checks_pass=True,
                  failing_checks=[],
              ),
              geoflex.evaluation.SingleEvaluationResult(
                  standard_error_absolute_effect=2.2,
                  standard_error_relative_effect=2.1,
                  coverage_absolute_effect=0.9,
                  coverage_relative_effect=0.99,
                  all_checks_pass=True,
                  failing_checks=[],
              ),
          ],
          "iROAS": [
              geoflex.evaluation.SingleEvaluationResult(
                  standard_error_absolute_effect=1.3,
                  standard_error_relative_effect=None,
                  coverage_absolute_effect=0.5,
                  coverage_relative_effect=None,
                  all_checks_pass=True,
                  failing_checks=[],
              ),
              geoflex.evaluation.SingleEvaluationResult(
                  standard_error_absolute_effect=2.2,
                  standard_error_relative_effect=None,
                  coverage_absolute_effect=0.7,
                  coverage_relative_effect=None,
                  all_checks_pass=False,
                  failing_checks=["something failed"],
              ),
          ],
          "CPiA __INVERTED__": [
              geoflex.evaluation.SingleEvaluationResult(
                  standard_error_absolute_effect=1.3,
                  standard_error_relative_effect=None,
                  coverage_absolute_effect=0.5,
                  coverage_relative_effect=None,
                  all_checks_pass=True,
                  failing_checks=[],
              ),
              geoflex.evaluation.SingleEvaluationResult(
                  standard_error_absolute_effect=2.2,
                  standard_error_relative_effect=None,
                  coverage_absolute_effect=0.7,
                  coverage_relative_effect=None,
                  all_checks_pass=False,
                  failing_checks=["something else failed"],
              ),
          ],
      },
  )


def test_experiment_design_evaluation_results_primary_metric_results_per_cell(
    mock_design_evaluation_results,
):
  assert mock_design_evaluation_results.primary_metric_results_per_cell == (
      mock_design_evaluation_results.all_metric_results_per_cell["revenue"]
  )


def test_experiment_design_evaluation_results_representiveness_score_is_least_representative(
    mock_design_evaluation_results,
):
  assert mock_design_evaluation_results.representiveness_score == min(
      mock_design_evaluation_results.representiveness_scores_per_cell
  )


def test_experiment_design_evaluation_results_all_metric_results_is_worst_case(
    mock_design_evaluation_results,
):

  assert mock_design_evaluation_results.all_metric_results[
      "revenue"
  ] == geoflex.evaluation.SingleEvaluationResult(
      standard_error_absolute_effect=2.2,
      standard_error_relative_effect=2.1,
      coverage_absolute_effect=0.4,
      coverage_relative_effect=0.5,
      all_checks_pass=True,
      failing_checks=[],
  )
  assert mock_design_evaluation_results.all_metric_results[
      "iROAS"
  ] == geoflex.evaluation.SingleEvaluationResult(
      standard_error_absolute_effect=2.2,
      standard_error_relative_effect=None,
      coverage_absolute_effect=0.5,
      coverage_relative_effect=None,
      all_checks_pass=False,
      failing_checks=["something failed"],
  )
  assert mock_design_evaluation_results.all_metric_results[
      "CPiA __INVERTED__"
  ] == geoflex.evaluation.SingleEvaluationResult(
      standard_error_absolute_effect=2.2,
      standard_error_relative_effect=None,
      coverage_absolute_effect=0.5,
      coverage_relative_effect=None,
      all_checks_pass=False,
      failing_checks=["something else failed"],
  )


def test_experiment_design_evaluation_results_primary_metric_results(
    mock_design_evaluation_results,
):
  assert (
      mock_design_evaluation_results.primary_metric_results
      == geoflex.evaluation.SingleEvaluationResult(
          standard_error_absolute_effect=2.2,
          standard_error_relative_effect=2.1,
          coverage_absolute_effect=0.4,
          coverage_relative_effect=0.5,
          all_checks_pass=True,
          failing_checks=[],
      )
  )


def test_invalid_experiment_design_results_has_none_for_all_properties():
  invalid_design_results = geoflex.evaluation.ExperimentDesignEvaluationResults(
      is_valid_design=False,
      primary_metric_name="revenue",
      alpha=0.1,
      alternative_hypothesis="two-sided",
      representiveness_scores_per_cell=None,
      all_metric_results_per_cell=None,
  )

  assert invalid_design_results.primary_metric_results is None
  assert invalid_design_results.primary_metric_results_per_cell is None
  assert invalid_design_results.all_metric_results is None
  assert invalid_design_results.representiveness_score is None
  assert invalid_design_results.get_mde(target_power=0.8, relative=False, aggregate_across_cells=True) == {}  # pylint: disable=g-explicit-bool-comparison


def test_get_mde_returns_correct_mde_for_relative_effects(
    mock_design_evaluation_results,
):
  assert mock_design_evaluation_results.get_mde(
      target_power=0.8, relative=True, aggregate_across_cells=True
  ) == {"revenue": 5.221597207101212, "iROAS": None, "CPiA": None}


def test_get_mde_returns_correct_mde_for_absolute_effects(
    mock_design_evaluation_results,
):
  assert mock_design_evaluation_results.get_mde(
      target_power=0.8, relative=False, aggregate_across_cells=True
  ) == {
      "revenue": 5.47024469315365,
      "iROAS": 5.47024469315365,
      "CPiA": 0.18280717885464282,
  }


def test_get_mde_returns_per_cell_mde_if_aggregate_across_cells_is_false(
    mock_design_evaluation_results,
):
  assert mock_design_evaluation_results.get_mde(
      target_power=0.8, relative=False, aggregate_across_cells=False
  ) == {
      "revenue": [3.2324173186817022, 5.47024469315365],
      "iROAS": [3.2324173186817022, 5.47024469315365],
      "CPiA": [0.3093659949847802, 0.18280717885464282],
  }


def test_get_summary_dict_returns_correct_dict(
    mock_design_evaluation_results,
):
  assert mock_design_evaluation_results.get_summary_dict(
      target_power=0.8, use_relative_effects_where_possible=True
  ) == {
      "failing_checks": ["something failed", "something else failed"],
      "all_checks_pass": False,
      "representiveness_score": 1.0,
      "primary_metric_failing_checks": [],
      "primary_metric_all_checks_pass": True,
      "primary_metric_standard_error": 2.1,
      "Relative MDE (revenue, primary metric)": 5.221597207101212,
      "MDE (iROAS)": 5.47024469315365,
      "MDE (CPiA)": 0.18280717885464282,
  }


def test_experiment_design_get_summary_dict_returns_correct_dict_without_evaluation_results(
    mock_design,
):
  assert mock_design.get_summary_dict() == {
      "design_id": mock_design.design_id,
      "experiment_budget": "-10%",
      "primary_metric": "revenue",
      "secondary_metrics": ["iROAS", "CPiA"],
      "methodology": "TestingMethodology",
      "runtime_weeks": 4,
      "n_cells": 3,
      "cell_volume_constraint": (
          "control: None, treatment_1: None, treatment_2: None"
      ),
      "effect_scope": "all_geos",
      "alpha": 0.1,
      "alternative_hypothesis": "two-sided",
      "random_seed": 0,
  }


def test_experiment_design_print_summary_dict_returns_correct_dict_with_evaluation_results(
    mock_design, mock_design_evaluation_results
):
  mock_design.evaluation_results = mock_design_evaluation_results
  assert mock_design.get_summary_dict() == {
      "design_id": mock_design.design_id,
      "experiment_budget": "-10%",
      "primary_metric": "revenue",
      "secondary_metrics": ["iROAS", "CPiA"],
      "methodology": "TestingMethodology",
      "runtime_weeks": 4,
      "n_cells": 3,
      "cell_volume_constraint": (
          "control: None, treatment_1: None, treatment_2: None"
      ),
      "effect_scope": "all_geos",
      "alpha": 0.1,
      "alternative_hypothesis": "two-sided",
      "random_seed": 0,
      "failing_checks": ["something failed", "something else failed"],
      "all_checks_pass": False,
      "representiveness_score": 1.0,
      "primary_metric_failing_checks": [],
      "primary_metric_all_checks_pass": True,
      "primary_metric_standard_error": 2.1,
      "Relative MDE (revenue, primary metric)": 5.221597207101212,
      "MDE (iROAS)": 5.47024469315365,
      "MDE (CPiA)": 0.18280717885464282,
  }


def test_experiment_design_print_summary_dict_returns_correct_dict_with_geo_assignment(
    mock_design,
):
  mock_design.geo_assignment = GeoAssignment(
      treatment=[{"US", "UK"}, {"CA"}],
      control={"FR", "AU"},
      exclude={"DE", "JP"},
  )
  assert mock_design.get_summary_dict() == {
      "design_id": mock_design.design_id,
      "experiment_budget": "-10%",
      "primary_metric": "revenue",
      "secondary_metrics": ["iROAS", "CPiA"],
      "methodology": "TestingMethodology",
      "runtime_weeks": 4,
      "n_cells": 3,
      "cell_volume_constraint": (
          "control: None, treatment_1: None, treatment_2: None"
      ),
      "effect_scope": "all_geos",
      "alpha": 0.1,
      "alternative_hypothesis": "two-sided",
      "random_seed": 0,
      "geo_assignment_control": "AU, FR",
      "geo_assignment_exclude": "DE, JP",
      "geo_assignment_treatment_1": "UK, US",
      "geo_assignment_treatment_2": "CA",
  }


def test_make_variation_returns_correct_design(
    mock_design, mock_design_evaluation_results
):
  mock_design.evaluation_results = mock_design_evaluation_results
  mock_design.geo_assignment = GeoAssignment(
      treatment=[{"US", "UK"}, {"CA"}],
      control={"FR", "AU"},
      exclude={"DE", "JP"},
  )

  variation = mock_design.make_variation(
      experiment_budget=ExperimentBudget(
          value=-0.5,
          budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
      ),
  )

  assert variation.design_id != mock_design.design_id
  assert variation.evaluation_results is None
  assert variation.geo_assignment is None
  assert variation.experiment_budget == ExperimentBudget(
      value=-0.5,
      budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
  )
  assert variation.geo_eligibility == mock_design.geo_eligibility
  assert variation.methodology == mock_design.methodology
  assert variation.runtime_weeks == mock_design.runtime_weeks
  assert variation.n_cells == mock_design.n_cells
  assert variation.alpha == mock_design.alpha
  assert variation.alternative_hypothesis == mock_design.alternative_hypothesis
  assert variation.random_seed == mock_design.random_seed


@pytest.mark.parametrize(
    "budget, expected_string",
    [
        (
            ExperimentBudget(
                value=-0.1,
                budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
            ),
            "-10%",
        ),
        (
            ExperimentBudget(
                value=-0.1,
                budget_type=ExperimentBudgetType.DAILY_BUDGET,
            ),
            "$-0.1 per day per cell",
        ),
        (
            ExperimentBudget(
                value=-0.1,
                budget_type=ExperimentBudgetType.TOTAL_BUDGET,
            ),
            "$-0.1 total per cell",
        ),
        (
            ExperimentBudget(
                value=[0.1, -0.1],
                budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
            ),
            "Cell 1: 10%, Cell 2: -10%",
        ),
        (
            ExperimentBudget(
                value=[0.1, -0.1],
                budget_type=ExperimentBudgetType.DAILY_BUDGET,
            ),
            "Cell 1: $0.1 per day, Cell 2: $-0.1 per day",
        ),
        (
            ExperimentBudget(
                value=[0.1, -0.1],
                budget_type=ExperimentBudgetType.TOTAL_BUDGET,
            ),
            "Cell 1: $0.1 total, Cell 2: $-0.1 total",
        ),
    ],
)
def test_experient_budget_string_representation_is_correct(
    budget, expected_string
):
  assert str(budget) == expected_string
