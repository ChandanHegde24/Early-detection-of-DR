import type { BiomarkerInput } from "@/lib/api/types";

type BiomarkerFormProps = {
  values: BiomarkerInput;
  onChange: (next: BiomarkerInput) => void;
  disabled?: boolean;
};

type FieldConfig = {
  key: keyof BiomarkerInput;
  label: string;
  min: number;
  max: number;
  step?: number;
};

const numericFields: FieldConfig[] = [
  { key: "age", label: "Age", min: 0, max: 120, step: 1 },
  { key: "bmi", label: "BMI", min: 10, max: 70, step: 0.1 },
  { key: "hba1c", label: "HbA1c (%)", min: 3, max: 20, step: 0.1 },
  { key: "blood_pressure_systolic", label: "Systolic BP", min: 60, max: 250, step: 1 },
  { key: "blood_pressure_diastolic", label: "Diastolic BP", min: 30, max: 150, step: 1 },
  { key: "cholesterol_total", label: "Total Cholesterol", min: 50, max: 500, step: 1 },
  { key: "cholesterol_hdl", label: "HDL", min: 10, max: 150, step: 1 },
  { key: "cholesterol_ldl", label: "LDL", min: 10, max: 400, step: 1 },
  { key: "triglycerides", label: "Triglycerides", min: 30, max: 1000, step: 1 },
  { key: "diabetes_duration_years", label: "Diabetes Duration (years)", min: 0, max: 80, step: 1 },
];

const inputClassName =
  "w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm outline-none transition focus:border-cyan-700 focus:ring-2 focus:ring-cyan-200 disabled:cursor-not-allowed disabled:bg-slate-100";

export function BiomarkerForm({ values, onChange, disabled = false }: BiomarkerFormProps) {
  function updateNumericField(key: keyof BiomarkerInput, rawValue: string) {
    const parsed = Number(rawValue);
    onChange({ ...values, [key]: Number.isFinite(parsed) ? parsed : 0 });
  }

  return (
    <section className="rounded-2xl border border-cyan-100 bg-white/90 p-5 shadow-sm">
      <h2 className="text-lg font-semibold text-slate-900">Clinical Biomarkers</h2>
      <p className="mt-1 text-sm text-slate-600">
        Enter patient-level metabolic and cardiovascular indicators used by the biomarker model.
      </p>

      <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
        {numericFields.map((field) => (
          <label key={field.key} className="space-y-1">
            <span className="text-xs font-medium uppercase tracking-wide text-slate-600">
              {field.label}
            </span>
            <input
              className={inputClassName}
              type="number"
              value={values[field.key]}
              min={field.min}
              max={field.max}
              step={field.step ?? 1}
              disabled={disabled}
              onChange={(event) => updateNumericField(field.key, event.target.value)}
            />
          </label>
        ))}

        <label className="space-y-1">
          <span className="text-xs font-medium uppercase tracking-wide text-slate-600">Smoking status</span>
          <select
            className={inputClassName}
            value={values.smoking_status}
            disabled={disabled}
            onChange={(event) => updateNumericField("smoking_status", event.target.value)}
          >
            <option value={0}>Never</option>
            <option value={1}>Former</option>
            <option value={2}>Current</option>
          </select>
        </label>

        <label className="space-y-1">
          <span className="text-xs font-medium uppercase tracking-wide text-slate-600">Family history of DR</span>
          <select
            className={inputClassName}
            value={values.family_history_dr}
            disabled={disabled}
            onChange={(event) => updateNumericField("family_history_dr", event.target.value)}
          >
            <option value={0}>No</option>
            <option value={1}>Yes</option>
          </select>
        </label>
      </div>
    </section>
  );
}
