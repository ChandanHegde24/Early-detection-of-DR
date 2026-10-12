"use client";

export type AnalysisMode =
  | "biomarker"
  | "image"
  | "unified";

type AnalysisModeSelectorProps = {
  mode: AnalysisMode;
  onChange: (mode: AnalysisMode) => void;
  disabled?: boolean;
};

const modes: {
  id: AnalysisMode;
  title: string;
  description: string;
  input: string;
  output: string;
}[] = [
  {
    id: "biomarker",
    title: "Biomarker Analysis",
    description:
      "Assess DR risk using patient clinical information.",
    input: "Clinical data only",
    output: "Biomarker risk",
  },
  {
    id: "image",
    title: "Retinal Image Analysis",
    description:
      "Analyze a fundus image using the CNN.",
    input: "Fundus image only",
    output: "DR Grade 0–4 + Grad-CAM",
  },
  {
    id: "unified",
    title: "Unified Screening",
    description:
      "Combine clinical and retinal evidence.",
    input: "Clinical data + fundus image",
    output: "Fused risk + DR grade",
  },
];

export function AnalysisModeSelector({
  mode,
  onChange,
  disabled = false,
}: AnalysisModeSelectorProps) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">

      <div className="mb-4">
        <h2 className="text-base font-bold text-slate-900">
          Choose Analysis Mode
        </h2>

        <p className="mt-1 text-xs text-slate-500">
          Select what you want RetinaGuard to analyze.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">

        {modes.map((item) => {
          const selected = mode === item.id;

          return (
            <button
              key={item.id}
              type="button"
              disabled={disabled}
              onClick={() => onChange(item.id)}
              className={`rounded-2xl border p-5 text-left transition-all ${
                selected
                  ? "border-cyan-500 bg-cyan-50 shadow-md ring-2 ring-cyan-500/10"
                  : "border-slate-200 bg-white hover:border-cyan-300 hover:bg-slate-50"
              } disabled:cursor-not-allowed disabled:opacity-60`}
            >

              <div className="flex items-start justify-between gap-3">

                <div>
                  <h3
                    className={`text-sm font-bold ${
                      selected
                        ? "text-cyan-900"
                        : "text-slate-900"
                    }`}
                  >
                    {item.title}
                  </h3>

                  <p className="mt-2 text-xs leading-relaxed text-slate-500">
                    {item.description}
                  </p>
                </div>

                <span
                  className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-xs font-bold ${
                    selected
                      ? "bg-cyan-600 text-white"
                      : "bg-slate-100 text-slate-500"
                  }`}
                >
                  {item.id === "biomarker"
                    ? "B"
                    : item.id === "image"
                      ? "C"
                      : "U"}
                </span>

              </div>

              <div className="mt-4 border-t border-slate-100 pt-3">

                <p className="text-[11px] text-slate-500">
                  <span className="font-semibold text-slate-700">
                    Input:
                  </span>{" "}
                  {item.input}
                </p>

                <p className="mt-1 text-[11px] text-slate-500">
                  <span className="font-semibold text-slate-700">
                    Output:
                  </span>{" "}
                  {item.output}
                </p>

              </div>

            </button>
          );
        })}

      </div>
    </section>
  );
}