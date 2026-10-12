"use client";

import { useState } from "react";
import { BiomarkerForm } from "@/components/forms/BiomarkerForm";
import { FundusUploader } from "@/components/upload/FundusUploader";
import { RiskCard } from "@/components/results/RiskCard";
import { ProbabilityChart } from "@/components/results/ProbabilityChart";
import { GradCamPanel } from "@/components/results/GradCamPanel";
import { usePrediction } from "@/hooks/usePrediction";
import { biomarkerDefaults } from "@/lib/validation/biomarker-schema";
import { downloadUnifiedReport } from "@/lib/api/client";
import type { BiomarkerInput } from "@/lib/api/types";

export default function ScreeningPage() {
  const [hasDiabetes, setHasDiabetes] = useState<boolean | null>(null);
  const [biomarkers, setBiomarkers] = useState<BiomarkerInput>(biomarkerDefaults);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const { loading, result, error, submitUnified, reset } = usePrediction();

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (hasDiabetes !== true || !imageFile) {
      return;
    }
    await submitUnified(imageFile, biomarkers);
  }

  async function handleDownloadReport() {
    if (!imageFile) {
      return;
    }
    await downloadUnifiedReport(imageFile, biomarkers);
  }

  function handleReset() {
    setHasDiabetes(null);
    setBiomarkers(biomarkerDefaults);
    setImageFile(null);
    reset();
  }

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top,_#dff6ff_0%,_#f7fcff_40%,_#eef6ff_100%)] px-4 py-8 sm:px-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <header className="rounded-3xl border border-cyan-200 bg-white/80 p-6 shadow-sm backdrop-blur-sm">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-700">Patient screening</p>
              <h1 className="mt-2 text-3xl font-bold text-slate-900 sm:text-4xl">Diabetic Retinopathy Check</h1>
            </div>
            <span className="rounded-full border border-cyan-200 bg-cyan-50 px-3 py-1 text-xs font-semibold text-cyan-800">
              Image + clinical AI
            </span>
          </div>
          <p className="mt-3 max-w-3xl text-sm text-slate-600 sm:text-base">
            Confirm diabetes status first. For a diabetic patient, upload a retinal fundus image and enter
            the clinical measurements required for a DR severity assessment.
          </p>
        </header>

        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Step 1</p>
              <h2 className="mt-1 text-xl font-semibold text-slate-900">Does the patient have diabetes?</h2>
              <p className="mt-1 text-sm text-slate-600">This determines whether retinal DR screening is needed.</p>
            </div>
            <div className="grid w-full grid-cols-2 gap-2 sm:w-auto" role="group" aria-label="Diabetes status">
              <button
                type="button"
                aria-pressed={hasDiabetes === true}
                onClick={() => {
                  setHasDiabetes(true);
                  reset();
                }}
                className={`rounded-xl border px-5 py-3 text-sm font-semibold transition ${
                  hasDiabetes === true
                    ? "border-cyan-700 bg-cyan-700 text-white"
                    : "border-slate-300 bg-white text-slate-700 hover:border-cyan-500 hover:bg-cyan-50"
                }`}
              >
                Yes
              </button>
              <button
                type="button"
                aria-pressed={hasDiabetes === false}
                onClick={() => {
                  setHasDiabetes(false);
                  setImageFile(null);
                  reset();
                }}
                className={`rounded-xl border px-5 py-3 text-sm font-semibold transition ${
                  hasDiabetes === false
                    ? "border-emerald-700 bg-emerald-700 text-white"
                    : "border-slate-300 bg-white text-slate-700 hover:border-emerald-500 hover:bg-emerald-50"
                }`}
              >
                No
              </button>
            </div>
          </div>
        </section>

        {hasDiabetes === false ? (
          <section className="rounded-2xl border border-emerald-200 bg-emerald-50 p-5 text-emerald-900">
            <h2 className="font-semibold">No diabetic retinopathy screening required from this workflow</h2>
            <p className="mt-1 text-sm text-emerald-800">
              The patient was marked as not having diabetes. Confirm this status clinically and continue routine care as appropriate.
            </p>
          </section>
        ) : null}

        {hasDiabetes === true ? (
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="flex items-center gap-3 rounded-2xl border border-cyan-200 bg-cyan-50 px-5 py-4 text-sm text-cyan-950">
            <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-cyan-700 font-bold text-white">2</span>
            <span>Upload a clear retinal fundus image and provide the patient biomarkers for evaluation.</span>
          </div>
          <div className="grid grid-cols-1 gap-6 xl:grid-cols-[1.3fr_0.9fr]">
            <BiomarkerForm values={biomarkers} onChange={setBiomarkers} disabled={loading} />
            <FundusUploader file={imageFile} onFileChange={setImageFile} disabled={loading} />
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <button
              type="submit"
              disabled={loading || !imageFile || hasDiabetes !== true}
              className="rounded-xl bg-cyan-700 px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-cyan-800 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading ? "Evaluating retinal image..." : "Evaluate diabetic retinopathy"}
            </button>
            <button
              type="button"
              disabled={loading}
              onClick={handleReset}
              className="rounded-xl border border-slate-300 bg-white px-5 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-60"
            >
              Reset
            </button>
            {!imageFile ? <p className="text-sm text-rose-700">Upload a fundus image to enable evaluation.</p> : null}
          </div>
        </form>
        ) : null}

        {error ? (
          <section className="rounded-2xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800">{error}</section>
        ) : null}

        {result ? (
          <section className="space-y-6">
            <div className="flex flex-wrap items-center gap-3">
              <button
                type="button"
                disabled={loading || !imageFile}
                onClick={handleDownloadReport}
                className="rounded-xl border border-cyan-300 bg-cyan-50 px-5 py-3 text-sm font-semibold text-cyan-900 transition hover:bg-cyan-100 disabled:cursor-not-allowed disabled:opacity-60"
              >
                Download Clinical PDF Report
              </button>
            </div>
            <RiskCard result={result} />
            <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
              <ProbabilityChart result={result} />
              <GradCamPanel result={result} />
            </div>
          </section>
        ) : null}
      </div>
    </main>
  );
}
