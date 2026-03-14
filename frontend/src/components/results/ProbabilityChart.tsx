import type { PredictionResponse } from "@/lib/api/types";

type ProbabilityChartProps = {
  result: PredictionResponse;
};

export function ProbabilityChart({ result }: ProbabilityChartProps) {
  return (
    <section className="rounded-2xl border border-cyan-100 bg-white/90 p-5 shadow-sm">
      <h3 className="text-base font-semibold text-slate-900">Per-grade Probabilities</h3>
      <div className="mt-4 space-y-3">
        {result.grade_probabilities.map((item) => {
          const percentage = Math.max(0, Math.min(100, item.probability * 100));
          const isTop = item.grade === result.predicted_grade;

          return (
            <div key={item.grade} className="space-y-1">
              <div className="flex items-center justify-between text-xs text-slate-700">
                <span className={isTop ? "font-semibold text-cyan-900" : "font-medium"}>{item.label}</span>
                <span>{percentage.toFixed(2)}%</span>
              </div>
              <div className="h-2.5 rounded-full bg-slate-200">
                <div
                  className={isTop ? "h-full rounded-full bg-cyan-700" : "h-full rounded-full bg-slate-400"}
                  style={{ width: `${percentage}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
