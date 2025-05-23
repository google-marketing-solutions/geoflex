/**
 * Metric description
 */
export interface Metric {
  column: string;
  cost_column: string;
  cost_per_metric: boolean;
  metric_per_cost: boolean;
  name: string;
}

/**
 * Server response with ExperimentDesign.
 */
export interface ExperimentDesignResponse {
  design_id: string;
  alpha: number;
  alternative_hypothesis: string;
  cell_volume_constraint: { constraint_type: string; values: Array<number | null> };
  effect_scope: string;
  evaluation_results;
  experiment_budget;
  geo_assignment;
  geo_eligibility;
  methodology: string;
  methodology_parameters;
  n_cells: number;
  primary_metric: Metric | undefined;
  random_seed: number;
  runtime_weeks: number;
  secondary_metrics: Metric[];
  mde: number; // As a percentage (e.g. 0.05 for 5%)
}

/**
 * Client-side description of a design.
 */
export interface ExperimentDesign {
  design_id: string;
  mde: number;
  runtime_weeks: number;
  methodology: string;
  methodology_parameters: Record<string, unknown>;
  isValid: boolean;
  parameters: {
    n_cells: number;
    primary_metric: Metric | undefined;
    secondary_metrics: Metric[];
    alpha: number;
    alternative_hypothesis: string;
    cell_volume_constraint: { constraint_type: string; values: Array<number | null> };
    effect_scope: string;
    random_seed: number;
  };
  evaluation_results?;
  groups: {
    Control: string[];
    Treatment?: string[];
    [key: string]: string[]; // For multi-cell tests (Test B, Test C, etc.)
  };
}

export interface ExperimentExploreResponse {
  designs: ExperimentDesignResponse[];
}
