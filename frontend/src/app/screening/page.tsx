"use client";

import { useState, type FormEvent } from "react";

import {
  AnalysisModeSelector,
  type AnalysisMode,
} from "@/components/ui/AnalysisModeSelector";

import { BiomarkerForm } from "@/components/forms/BiomarkerForm";
import { FundusUploader } from "@/components/upload/FundusUploader";

import { RiskCard } from "@/components/results/RiskCard";
import { ProbabilityChart } from "@/components/results/ProbabilityChart";
import { GradCamPanel } from "@/components/results/GradCamPanel";

import {
  predictBiomarker,
  predictImageOnly,
  predictUnified,
  downloadUnifiedReport,
} from "@/lib/api/client";

import { biomarkerDefaults } from "@/lib/validation/biomarker-schema";

import type {
  BiomarkerInput,
  PredictionResponse,
} from "@/lib/api/types";

export default function ScreeningPage() {
  /* ------------------------------------------------------------
     State
  ------------------------------------------------------------ */

  const [analysisMode, setAnalysisMode] =
    useState<AnalysisMode>("unified");

  const [biomarkers, setBiomarkers] =
    useState<BiomarkerInput>(biomarkerDefaults);

  const [imageFile, setImageFile] =
    useState<File | null>(null);

  const [loading, setLoading] =
    useState(false);

  const [result, setResult] =
    useState<PredictionResponse | null>(null);

  const [error, setError] =
    useState<string | null>(null);

  /* ------------------------------------------------------------
     Mode change
  ------------------------------------------------------------ */

  function handleModeChange(mode: AnalysisMode) {
    setAnalysisMode(mode);

    // Clear old result when changing analysis type.
    setResult(null);
    setError(null);
  }

  /* ------------------------------------------------------------
     Prediction
  ------------------------------------------------------------ */

  async function handleSubmit(
    event: FormEvent<HTMLFormElement>
  ) {
    event.preventDefault();

    setError(null);
    setResult(null);
    setLoading(true);

    try {
      let response: PredictionResponse;

      /* ---------------- Biomarker Only ---------------- */

      if (analysisMode === "biomarker") {
        response = await predictBiomarker(biomarkers);
      }

      /* ---------------- CNN / Image Only ---------------- */

      else if (analysisMode === "image") {
        if (!imageFile) {
          throw new Error(
            "Please upload a retinal fundus image."
          );
        }

        response = await predictImageOnly(imageFile);
      }

      /* ---------------- Unified / Fusion ---------------- */

      else {
        if (!imageFile) {
          throw new Error(
            "Please upload a retinal fundus image for unified screening."
          );
        }

        response = await predictUnified(
          imageFile,
          biomarkers
        );
      }

      setResult(response);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Prediction failed. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  }

  /* ------------------------------------------------------------
     Reset
  ------------------------------------------------------------ */

  function handleReset() {
    setAnalysisMode("unified");
    setBiomarkers(biomarkerDefaults);
    setImageFile(null);
    setResult(null);
    setError(null);
  }

  /* ------------------------------------------------------------
     PDF Report
     Only Unified Screening currently supports the full
     clinical report because it requires image + biomarkers.
  ------------------------------------------------------------ */

  async function handleDownloadReport() {
    if (analysisMode !== "unified") {
      return;
    }

    if (!imageFile) {
      setError(
        "Please upload a retinal fundus image before generating the report."
      );
      return;
    }

    try {
      setError(null);

      await downloadUnifiedReport(
        imageFile,
        biomarkers
      );
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Failed to generate the clinical report.");
      }
    }
  }

  /* ------------------------------------------------------------
     Dynamic page information
  ------------------------------------------------------------ */

  const pageInformation: Record<
    AnalysisMode,
    {
      title: string;
      description: string;
      button: string;
    }
  > = {
    biomarker: {
      title: "Biomarker Analysis",
      description:
        "Assess diabetic retinopathy risk using patient clinical and metabolic information only.",
      button: "Run Biomarker Analysis",
    },

    image: {
      title: "Retinal Image Analysis",
      description:
        "Analyze a retinal fundus image using the CNN to estimate the five-stage diabetic retinopathy severity.",
      button: "Analyze Retinal Image",
    },

    unified: {
      title: "Unified Screening",
      description:
        "Combine clinical biomarkers and retinal image evidence using the multimodal fusion pipeline.",
      button: "Run Unified Screening",
    },
  };

  const currentMode = pageInformation[analysisMode];

  /* ------------------------------------------------------------
     Render
  ------------------------------------------------------------ */

  const biomarkerNoDrProbability =
  result &&
  analysisMode === "biomarker" &&
  result.grade_probabilities?.[0]
    ? result.grade_probabilities[0].probability
    : null;

  const biomarkerHasDrProbability =
    result &&
    analysisMode === "biomarker" &&
    result.grade_probabilities?.[1]
     ? result.grade_probabilities[1].probability
     : null;

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top,_#dff6ff_0%,_#f7fcff_40%,_#eef6ff_100%)] px-4 py-8 sm:px-8">
      <div className="mx-auto max-w-7xl space-y-6">

        {/* ======================================================
            Header
        ====================================================== */}

        <header className="rounded-3xl border border-cyan-200 bg-white/80 p-6 shadow-sm backdrop-blur-sm">

          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-700">
            RetinaGuard AI Clinical Suite
          </p>

          <h1 className="mt-2 text-3xl font-bold text-slate-900 sm:text-4xl">
            Diabetic Retinopathy Screening
          </h1>

          <p className="mt-3 max-w-3xl text-sm text-slate-600 sm:text-base">
            Choose the type of analysis you want to perform:
            clinical biomarker analysis, retinal image analysis,
            or combined unified screening.
          </p>
        </header>

        {/* ======================================================
            Analysis Mode Selector
        ====================================================== */}

        <AnalysisModeSelector
          mode={analysisMode}
          onChange={handleModeChange}
          disabled={loading}
        />

        {/* ======================================================
            Selected Mode Description
        ====================================================== */}

        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">

          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">

            <div>
              <h2 className="text-lg font-bold text-slate-900">
                {currentMode.title}
              </h2>

              <p className="mt-1 text-sm text-slate-500">
                {currentMode.description}
              </p>
            </div>

            <span className="w-fit rounded-full border border-cyan-200 bg-cyan-50 px-3 py-1 text-xs font-semibold text-cyan-700">
              {analysisMode === "biomarker"
                ? "Clinical Data"
                : analysisMode === "image"
                  ? "Fundus Image"
                  : "Multimodal"}
            </span>

          </div>
        </section>

        {/* ======================================================
            Input Form
        ====================================================== */}

        <form
          onSubmit={handleSubmit}
          className="space-y-6"
        >

          {/* ----------------------------------------------------
              Biomarker Input
          ---------------------------------------------------- */}

          {analysisMode !== "image" && (
            <section>
              <BiomarkerForm
                values={biomarkers}
                onChange={setBiomarkers}
                disabled={loading}
              />
            </section>
          )}

          {/* ----------------------------------------------------
              Fundus Image Input
          ---------------------------------------------------- */}

          {analysisMode !== "biomarker" && (
            <section>
              <FundusUploader
                file={imageFile}
                onFileChange={setImageFile}
                disabled={loading}
              />
            </section>
          )}

          {/* ----------------------------------------------------
              Action Buttons
          ---------------------------------------------------- */}

          <div className="flex flex-wrap items-center gap-3">

            <button
              type="submit"
              disabled={
                loading ||
                (analysisMode !== "biomarker" && !imageFile)
              }
              className="rounded-xl bg-cyan-700 px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-cyan-800 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading
                ? "Running AI analysis..."
                : currentMode.button}
            </button>

            <button
              type="button"
              disabled={loading}
              onClick={handleReset}
              className="rounded-xl border border-slate-300 bg-white px-5 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-60"
            >
              Reset
            </button>

            {analysisMode !== "biomarker" &&
              !imageFile && (
                <p className="text-sm text-rose-700">
                  Upload a fundus image to enable this analysis.
                </p>
              )}

          </div>
        </form>

        {/* ======================================================
            Error
        ====================================================== */}

        {error && (
          <section className="rounded-2xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800">
            <p className="font-semibold">
              Analysis Error
            </p>

            <p className="mt-1">
              {error}
            </p>
          </section>
        )}

        {/* ======================================================
            Results
        ====================================================== */}

        {result && (
          <section className="space-y-6">

            {/* --------------------------------------------------
                Biomarker-only Result
            -------------------------------------------------- */}

            {analysisMode === "biomarker" && (
  <section className="space-y-6">

    {/* =====================================================
        Biomarker Analysis Result
    ====================================================== */}

    <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">

      <div className="flex flex-col gap-4 border-b border-slate-100 pb-5 sm:flex-row sm:items-start sm:justify-between">

        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-cyan-700">
            Biomarker Analysis Result
          </p>

          <h2 className="mt-2 text-2xl font-bold text-slate-900">
            Clinical Risk Assessment
          </h2>

          <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-500">
            This result was generated using patient clinical and
            metabolic information only. No retinal fundus image was
            used for this prediction.
          </p>
        </div>

        <span className="w-fit rounded-full border border-cyan-200 bg-cyan-50 px-3 py-1.5 text-xs font-semibold text-cyan-700">
          Biomarker Only
        </span>

      </div>

      {/* -----------------------------------------------------
          Main Prediction
      ------------------------------------------------------ */}

      <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-3">

        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-5">

          <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Prediction
          </p>

          <p
            className={`mt-2 text-2xl font-extrabold ${
              result.predicted_label === "Has DR"
                ? "text-rose-600"
                : "text-emerald-600"
            }`}
          >
            {result.predicted_label}
          </p>

          <p className="mt-1 text-xs text-slate-500">
            Based on clinical biomarkers
          </p>

        </div>

        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-5">

          <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            DR Risk Probability
          </p>

          <p className="mt-2 text-2xl font-extrabold text-cyan-700">
            {(result.risk_score * 100).toFixed(1)}%
          </p>

          <p className="mt-1 text-xs text-slate-500">
            Probability of Has DR
          </p>

        </div>

        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-5">

          <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Screening Tier
          </p>

          <p className="mt-2 text-2xl font-extrabold text-slate-900">
            {result.screening_tier}
          </p>

          <p className="mt-1 text-xs text-slate-500">
            Clinical risk classification
          </p>

        </div>

      </div>

      {/* -----------------------------------------------------
          Binary Probability Breakdown
      ------------------------------------------------------ */}

      <div className="mt-6 rounded-2xl border border-slate-200 bg-white p-5">

        <div className="mb-4">
          <h3 className="text-sm font-bold text-slate-900">
            Biomarker Prediction Probabilities
          </h3>

          <p className="mt-1 text-xs text-slate-500">
            Binary output from the clinical biomarker model.
          </p>
        </div>

        <div className="space-y-4">

          {/* No DR */}

          <div>

            <div className="mb-1.5 flex items-center justify-between">

              <span className="text-xs font-semibold text-slate-700">
                No DR
              </span>

              <span className="text-xs font-bold text-slate-900">
                {biomarkerNoDrProbability !== null
                  ? `${(biomarkerNoDrProbability * 100).toFixed(2)}%`
                  : "N/A"}
              </span>

            </div>

            <div className="h-3 overflow-hidden rounded-full bg-slate-100">

              <div
                className="h-full rounded-full bg-emerald-500 transition-all duration-700"
                style={{
                  width: `${
                    biomarkerNoDrProbability !== null
                      ? biomarkerNoDrProbability * 100
                      : 0
                  }%`,
                }}
              />

            </div>

          </div>

          {/* Has DR */}

          <div>

            <div className="mb-1.5 flex items-center justify-between">

              <span className="text-xs font-semibold text-slate-700">
                Has DR
              </span>

              <span className="text-xs font-bold text-slate-900">
                {biomarkerHasDrProbability !== null
                  ? `${(biomarkerHasDrProbability * 100).toFixed(2)}%`
                  : "N/A"}
              </span>

            </div>

            <div className="h-3 overflow-hidden rounded-full bg-slate-100">

              <div
                className="h-full rounded-full bg-rose-500 transition-all duration-700"
                style={{
                  width: `${
                    biomarkerHasDrProbability !== null
                      ? biomarkerHasDrProbability * 100
                      : 0
                  }%`,
                }}
              />

            </div>

          </div>

        </div>

      </div>

      {/* -----------------------------------------------------
          Clinical Recommendation
      ------------------------------------------------------ */}

      {result.baseline_recommendation && (
        <div className="mt-5 rounded-2xl border border-cyan-100 bg-cyan-50 p-5">

          <p className="text-xs font-semibold uppercase tracking-wider text-cyan-700">
            Clinical Screening Guidance
          </p>

          <p className="mt-2 text-sm leading-relaxed text-cyan-950">
            {result.baseline_recommendation}
          </p>

        </div>
      )}

      {/* -----------------------------------------------------
          Model Information
      ------------------------------------------------------ */}

      <div className="mt-5 flex flex-wrap gap-2">

        <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-[11px] font-semibold text-slate-600">
          Model: {result.model_used}
        </span>

        <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-[11px] font-semibold text-slate-600">
          Input: Clinical biomarkers only
        </span>

        <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-[11px] font-semibold text-slate-600">
          Output: Binary DR risk
        </span>

      </div>

    </section>

  </section>
)}

            {/* --------------------------------------------------
                CNN-only Result
            -------------------------------------------------- */}

            {analysisMode === "image" && (
              <>
                <RiskCard result={result} />

                <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
                  <ProbabilityChart result={result} />
                  <GradCamPanel result={result} />
                </div>
              </>
            )}

            {/* --------------------------------------------------
                Unified Result
            -------------------------------------------------- */}

            {analysisMode === "unified" && (
              <>
                <div className="flex flex-wrap items-center gap-3">

                  <button
                    type="button"
                    disabled={!imageFile}
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
              </>
            )}

          </section>
        )}

      </div>
    </main>
  );
}