/**
 * The JSON contract emitted by `python -m ppict.cli analyze`.
 *
 * This mirrors `Report.as_dict()` in `src/ppict/report.py`. If the Python shape
 * changes, `CONTRACT_VERSION` changes there and this file changes with it. The
 * calculator does no statistics of its own, so every number below arrives from
 * the package.
 */

export const EXPECTED_CONTRACT = 1;

export type SourceTag = "measured" | "assumed";

export interface MethodRow {
  method: string;
  estimate: number;
  se: number;
  ci_low: number;
  ci_high: number;
  ci_width: number;
  p_value: number;
  lam: number | null;
  lambda_control: number | null;
  lambda_treated: number | null;
  n_treated: number;
  n_control: number;
  n_parameters: number;
  width_ratio: number | null;
}

export interface DatasetInfo {
  source: string;
  sha256: string;
  n_trial: number;
  n_treated: number;
  n_control: number;
  n_external: number;
  n_dropped: number;
  notes: string[];
}

export interface Measured {
  score_r2_control: number | null;
  score_r2_treated: number | null;
  outcome_sd: number | null;
  effect_unadjusted: number | null;
}

export interface Analysis {
  dataset: DatasetInfo;
  alpha: number;
  methods: MethodRow[];
  primary_method: string;
  primary_width_ratio: number | null;
  measured: Measured;
  unavailable: { method: string; reason: string }[];
}

export interface PlanInputs {
  cost_per_patient: number;
  power: number;
  r2: number;
  n_total: number;
  effect: number;
  outcome_sd: number;
  patients_per_month: number;
  alpha: number;
}

export interface Plan {
  inputs: PlanInputs;
  sources: Record<string, SourceTag>;
  variance_unadjusted: number | null;
  variance_adjusted: number | null;
  n_required_unadjusted: number | null;
  n_required_adjusted: number | null;
  patients_saved: number | null;
  cost_saved: number | null;
  months_saved: number | null;
  width_ratio: number | null;
  ci_narrowing: number | null;
  equivalent_unadjusted_n: number | null;
  notes: string[];
}

export interface Report {
  contract: number;
  generated_at: string;
  ppict_version: string;
  command: string;
  analysis: Analysis;
  plan: Plan;
  sap_tex: string;
  sap_filename: string;
}

export interface ApiErrorBody {
  error: string;
  detail?: string;
}

