export type GradeProbability = {
  grade: number;
  label: string;
  probability: number;
};

export type BiomarkerInput = {
  age: number;
  bmi: number;
  hba1c: number;
  blood_pressure_systolic: number;
  blood_pressure_diastolic: number;
  cholesterol_total: number;
  cholesterol_hdl: number;
  cholesterol_ldl: number;
  triglycerides: number;
  diabetes_duration_years: number;
  smoking_status: number;
  family_history_dr: number;
};

export type PredictionResponse = {
  predicted_grade: number;
  predicted_label: string;
  risk_score: number;
  screening_tier: string;
  grade_probabilities: GradeProbability[];
  model_used: string;
  baseline_clinical_score: number | null;
  baseline_recommendation: string | null;
  baseline_factor_breakdown: Record<string, number> | null;
  grad_cam_available: boolean;
  grad_cam_heatmap: string | null;
  grad_cam_overlay: string | null;
};

export type BatchItemResult = {
  patient_id: string;
  age: number;
  hba1c: number;
  blood_pressure_systolic: number;
  blood_pressure_diastolic: number;
  bmi: number;
  diabetes_duration_years: number;
  risk_score: number;
  screening_tier: "Urgent" | "Moderate" | "Low Risk" | string;
  predicted_grade: number;
  predicted_label: string;
  baseline_recommendation?: string | null;
};

export type BatchPredictionResponse = {
  total_patients: number;
  urgent_count: number;
  moderate_count: number;
  low_risk_count: number;
  avg_risk_score: number;
  results: BatchItemResult[];
};

