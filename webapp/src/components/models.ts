export interface ExperimentDesign {
  design_id: string;
  methodology: string;
  power: number; // As a percentage (0-100)
  mde: number; // As a percentage
  duration: number; // Runtime in weeks
  parameters: {
    methodology: string;
    experiment_type: string;
    runtime_weeks: number;
    n_cells: number;
    alpha: number;
    alternative_hypothesis: string;
    primary_metric?: string;
    //[key: string]: any; // For any additional parameters
  };
  groups: {
    Control: string[];
    Test: string[];
    [key: string]: string[]; // For multi-cell tests (Test B, Test C, etc.)
  };
}

export interface ExperimentExploreResponse {
  designs: ExperimentDesign[];
}
