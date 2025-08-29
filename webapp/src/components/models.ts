/**
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * A comprehensive interface to represent any metric type on the client-side.
 * This allows for a unified structure in UI components.
 */
export interface Metric {
  // Common properties
  /** The name of the metric. */
  name: string;
  /** The type of the metric. */
  type: 'custom' | 'iroas' | 'cpia';

  /** The column name for a custom metric. */
  column?: string;

  /** The column name for the cost in a custom metric. */
  cost_column?: string;
  /** Whether to calculate metric per cost. */
  metric_per_cost?: boolean;
  /** Whether to calculate cost per metric. */
  cost_per_metric?: boolean;

  /** The column name for the return value in an iROAS metric. */
  return_column?: string;

  /** The column name for conversions in a CPiA metric. */
  conversions_column?: string;
}

/**
 * A union type representing any possible metric configuration that can be sent to the backend.
 * The backend will parse this based on the 'type' and other properties.
 */
export type AnyMetric = string | Metric;

/**
 * Represents the evaluation results for a single metric.
 */
export interface EvaluationMetricResult {
  /** The standard error of the absolute effect. */
  standard_error_absolute_effect: number;
  /** The standard error of the relative effect. */
  standard_error_relative_effect: number;
  /** The coverage of the absolute effect. */
  coverage_absolute_effect: number;
  /** The coverage of the relative effect. */
  coverage_relative_effect: number;
  /** The empirical power of the metric. */
  empirical_power: number;
  /** Whether all checks passed for this metric. */
  all_checks_pass: boolean;
  /** A list of checks that failed. */
  failing_checks: string[];
}

/**
 * Represents the overall results of a design evaluation (power analysis).
 */
export interface EvaluationResults {
  /** The name of the primary metric. */
  primary_metric_name: string;
  /** The alpha value (significance level) for the evaluation. */
  alpha: number;
  /** The alternative hypothesis for the evaluation. */
  alternative_hypothesis: string;
  /** The evaluation results for all metrics, per cell. */
  all_metric_results_per_cell: {
    [metricName: string]: EvaluationMetricResult[];
  };
  /** The representativeness scores per cell. */
  representativeness_scores_per_cell: number[];
  /** The actual cell volumes. */
  actual_cell_volumes: null | number[];
  /** Any other errors that occurred during evaluation. */
  other_errors: string[];
  /** Whether the design is robust. This will be True if the design
   * does not have any errors and has sufficient simulations.
   * it will be False if it does not have any errors and has sufficient simulations,
   * and it will be None if it does not have sufficient simulations. */
  is_robust_design: boolean | null;
  /** Any warnings generated during evaluation. */
  warnings: string[];
  /** Whether a sufficient number of simulations were run. */
  sufficient_simulations: boolean;
  /** A list of checks that failed during evaluation. */
  failing_checks?: string[];
}

/**
 * Represents the budget for an experiment.
 */
export interface ExperimentBudget {
  /** The value of the budget. */
  value: number;
  /** The type of the budget (e.g., 'cost', 'impressions'). */
  budget_type: string;
}

/**
 * Represents the assignment of geos to control and treatment groups.
 */
export interface GeoAssignment {
  /** The geos assigned to the control group. */
  control: string[];
  /** The geos assigned to the treatment groups. */
  treatment: string[][];
  /** The geos to exclude from the experiment. */
  exclude: string[];
  /** All available geos. */
  all_geos: string[];
  /** Whether the geo assignment is flexible. */
  flexible: boolean;
}

/**
 * Represents the eligible geos for an experiment.
 */
export interface GeoEligibility {
  /** The eligible geos for the control group. */
  control: string[];
  /** The eligible geos for the treatment groups. */
  treatment: string[][];
  /** The geos to exclude from eligibility. */
  exclude: string[];
  /** All available geos for eligibility. */
  all_geos: string[];
  /** Whether the geo eligibility is flexible. */
  flexible: boolean;
}

/**
 * Client-side description of a design, matching the server's DesignSummary.
 */
export interface ExperimentDesign {
  /** The unique identifier for the design. */
  design_id: string;
  /** The alpha value (significance level) for the experiment. */
  alpha: number;
  /** The alternative hypothesis for the experiment. */
  alternative_hypothesis: string;
  /** The constraint on cell volume. */
  cell_volume_constraint: {
    /** The type of constraint. */
    constraint_type: string;
    /** The values for the constraint. */
    values: Array<number | null>;
    /** The metric column for the constraint. */
    metric_column: string | null;
  };
  /** The scope of the effect being measured. */
  effect_scope: string;
  /** The results of the design evaluation. */
  evaluation_results: EvaluationResults;
  /** The budget for the experiment. */
  experiment_budget: ExperimentBudget;
  /** The assignment of geos to control and treatment groups. */
  geo_assignment: GeoAssignment;
  /** The eligible geos for the experiment. */
  geo_eligibility: GeoEligibility;
  /** The methodology used for the experiment design. */
  methodology: string;
  /** The parameters for the chosen methodology. */
  methodology_parameters: Record<string, unknown>;
  /** The number of cells (treatment groups + control). */
  n_cells: number;
  /** The primary metric for the experiment. */
  primary_metric: AnyMetric | undefined;
  /** The random seed used for the design. */
  random_seed: number;
  /** The duration of the experiment in weeks. */
  runtime_weeks: number;
  /** The secondary metrics for the experiment. */
  secondary_metrics: AnyMetric[];
}

/**
 * Represents a saved experiment design along with its metadata.
 */
export interface SavedDesign {
  /** The experiment design. */
  design: ExperimentDesign;
  /** The name of the design. */
  name?: string;
  /** The Minimum Detectable Effect (MDE) for the design. */
  mde: number;
  /** The name of the datasource used for the design. */
  datasource_name: string;
  /** The timestamp when the design was saved. */
  timestamp: string;
  /** The user who saved the design. */
  user?: string;
  /** The timestamp when the datasource was last updated. */
  datasource_updated?: string;
  /** The start date of the data used for the design. */
  start_date?: string;
  /** The end date of the data used for the design. */
  end_date?: string;
}

/**
 * Represents the response from the server when exploring saved designs.
 */
export interface ExperimentExploreResponse {
  /** A list of saved designs. */
  designs: SavedDesign[];
  /** A list of log entries from the backend. */
  logs: LogEntry[];
}

/**
 * Represents a single log entry from the backend.
 */
export interface LogEntry {
  /** The timestamp of the log entry. */
  timestamp: string;
  /** The log level (e.g., 'INFO', 'WARNING', 'ERROR'). */
  level: string;
  /** The name of the logger that created the entry. */
  logger_name: string;
  /** The log message. */
  message: string;
  /** The module where the log entry was created. */
  module: string;
  /** The function where the log entry was created. */
  function: string;
  /** The line number where the log entry was created. */
  line_number: number;
}

/**
 * Represents the result of validating a datasource against a design.
 */
export interface ValidationResult {
  metrics: {
    missing: string[];
  };
  geoUnits: {
    designOnly: string[];
    dataSourceOnly: string[];
  };
  dates: {
    valid: boolean;
    required: {
      start: string;
      end: string;
    };
    actual: {
      start: string;
      end: string;
    };
  } | null;
}
