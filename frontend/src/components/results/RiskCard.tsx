import type { PredictionResponse } from "@/lib/api/types";
import { cn } from "@/lib/utils/cn";

type RiskCardProps = {
  result: PredictionResponse;
};

const tierStyles: Record<string, string> = {
  Urgent: "border-rose-300 bg-rose-50 text-rose-900",
  Moderate: "border-amber-300 bg-amber-50 text-amber-900",
  "Low Risk": "border-emerald-300 bg-emerald-50 text-emerald-900",
};

export function RiskCard({ result }: RiskCardProps) {
  const tierClass = tierStyles[result.screening_tier] ?? "border-slate-300 bg-slate-50 text-slate-900";
  const baselineScore = result.baseline_clinical_score;

  return (
    <section className={cn("rounded-2xl border p-5 shadow-sm", tierClass)}>
      <p className="text-xs font-semibold uppercase tracking-wide">Screening tier</p>
      <p className="mt-1 text-2xl font-bold">{result.screening_tier}</p>

      <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
        <Metric label="Predicted grade" value={`${result.predicted_grade} - ${result.predicted_label}`} />
        <Metric label="Risk score" value={result.risk_score.toFixed(3)} />
        <Metric label="Model" value={result.model_used} />
      </div>

      {baselineScore !== null ? (
        <div className="mt-4 rounded-xl border border-current/20 bg-white/70 p-3">
          <p className="text-[11px] font-medium uppercase tracking-wide opacity-70">Stage-1 clinical baseline</p>
          <p className="mt-1 text-sm font-semibold">Score: {baselineScore.toFixed(3)}</p>
          <p className="mt-1 text-sm leading-relaxed">{result.baseline_recommendation ?? "No recommendation available."}</p>
        </div>
      ) : null}
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-current/20 bg-white/60 p-3">
      <p className="text-[11px] font-medium uppercase tracking-wide opacity-70">{label}</p>
      <p className="mt-1 text-sm font-semibold leading-snug">{value}</p>
    </div>
  );
}
