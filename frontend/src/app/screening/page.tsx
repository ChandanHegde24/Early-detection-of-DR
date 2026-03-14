"use client";

import { useState } from "react";
import { BiomarkerForm } from "@/components/forms/BiomarkerForm";
import { FundusUploader } from "@/components/upload/FundusUploader";
import { RiskCard } from "@/components/results/RiskCard";
import { ProbabilityChart } from "@/components/results/ProbabilityChart";
import { GradCamPanel } from "@/components/results/GradCamPanel";
import { usePrediction } from "@/hooks/usePrediction";
import { biomarkerDefaults } from "@/lib/validation/biomarker-schema";
import type { BiomarkerInput } from "@/lib/api/types";

export default function ScreeningPage() {
  const [biomarkers, setBiomarkers] = useState<BiomarkerInput>(biomarkerDefaults);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const { loading, result, error, submitUnified, reset } = usePrediction();

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!imageFile) {
      return;
    }
    await submitUnified(imageFile, biomarkers);
  }

  function handleReset() {
    setBiomarkers(biomarkerDefaults);
    setImageFile(null);
    reset();
  }

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top,_#dff6ff_0%,_#f7fcff_40%,_#eef6ff_100%)] px-4 py-8 sm:px-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <header className="rounded-3xl border border-cyan-200 bg-white/80 p-6 shadow-sm backdrop-blur-sm">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-700">Unified DR AI Workflow</p>
          <h1 className="mt-2 text-3xl font-bold text-slate-900 sm:text-4xl">New Diabetic Retinopathy Screening</h1>
          <p className="mt-3 max-w-3xl text-sm text-slate-600 sm:text-base">
            Combine clinical biomarkers and a retinal fundus image to produce a fused severity grade, risk
            score, triage tier, and Grad-CAM interpretability maps.
          </p>
        </header>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 gap-6 xl:grid-cols-[1.3fr_0.9fr]">
            <BiomarkerForm values={biomarkers} onChange={setBiomarkers} disabled={loading} />
            <FundusUploader file={imageFile} onFileChange={setImageFile} disabled={loading} />
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <button
              type="submit"
              disabled={loading || !imageFile}
              className="rounded-xl bg-cyan-700 px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-cyan-800 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading ? "Running unified prediction..." : "Run unified screening"}
            </button>
            <button
              type="button"
              disabled={loading}
              onClick={handleReset}
              className="rounded-xl border border-slate-300 bg-white px-5 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-60"
            >
              Reset
            </button>
            {!imageFile ? <p className="text-sm text-rose-700">Upload a fundus image to enable prediction.</p> : null}
          </div>
        </form>

        {error ? (
          <section className="rounded-2xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800">{error}</section>
        ) : null}

        {result ? (
          <section className="space-y-6">
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
