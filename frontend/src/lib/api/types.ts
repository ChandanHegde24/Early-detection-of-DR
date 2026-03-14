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
  grad_cam_available: boolean;
  grad_cam_heatmap: string | null;
  grad_cam_overlay: string | null;
};
