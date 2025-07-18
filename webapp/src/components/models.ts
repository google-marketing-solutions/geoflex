/**
 * A comprehensive interface to represent any metric type on the client-side.
 * This allows for a unified structure in UI components.
 */
export interface Metric {
  // Common properties
  name: string;
  type: 'custom' | 'iroas' | 'cpia';

  // For custom metrics
  column?: string;

  // For custom metrics
  cost_column?: string;
  metric_per_cost?: boolean;
  cost_per_metric?: boolean;

  // For iROAS
  return_column?: string;

  // For CPiA
  conversions_column?: string;
}

/**
 * A union type representing any possible metric configuration that can be sent to the backend.
 * The backend will parse this based on the 'type' and other properties.
 */
export type AnyMetric = string | Metric;

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
  primary_metric: AnyMetric | undefined;
  random_seed: number;
  runtime_weeks: number;
  secondary_metrics: AnyMetric[];
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
    primary_metric: AnyMetric | undefined;
    secondary_metrics: AnyMetric[];
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
