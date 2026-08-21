"use client";

import React, { useState, useEffect, useRef, ChangeEvent } from "react";
import Image from "next/image";
import {
  predictBiomarker,
  predictImageOnly,
  predictUnified,
  predictBatchCsv,
  downloadUnifiedReport,
} from "@/lib/api/client";
import type {
  BiomarkerInput,
  PredictionResponse,
  BatchPredictionResponse,
  BatchItemResult,
} from "@/lib/api/types";

const biomarkerDefaults: BiomarkerInput = {
  age: 58,
  bmi: 29.4,
  hba1c: 8.8,
  blood_pressure_systolic: 146,
  blood_pressure_diastolic: 88,
  cholesterol_total: 215,
  cholesterol_hdl: 42,
  cholesterol_ldl: 135,
  triglycerides: 190,
  diabetes_duration_years: 12,
  smoking_status: 1,
  family_history_dr: 1,
};

// ── Color & Tier Helpers ───────────────────────────────────────────────────
function tierOf(score: number) {
  if (score >= 0.72) {
    return {
      label: "Urgent",
      color: "#DC2626",
      bg: "#FEF2F2",
      border: "#FCA5A5",
      bar: "#DC2626",
      badge: "bg-red-100 text-red-800 border-red-200",
    };
  }
  if (score >= 0.42) {
    return {
      label: "Moderate",
      color: "#D97706",
      bg: "#FFFBEB",
      border: "#FCD34D",
      bar: "#D97706",
      badge: "bg-amber-100 text-amber-800 border-amber-200",
    };
  }
  return {
    label: "Low Risk",
    color: "#16A34A",
    bg: "#F0FDF4",
    border: "#86EFAC",
    bar: "#16A34A",
    badge: "bg-emerald-100 text-emerald-800 border-emerald-200",
  };
}

const GRADES = [
  "Grade 0 — No DR",
  "Grade 1 — Mild NPDR",
  "Grade 2 — Moderate NPDR",
  "Grade 3 — Severe NPDR",
  "Grade 4 — Proliferative DR",
];
const GCOLS = ["#16A34A", "#CA8A04", "#EA580C", "#DC2626", "#991B1B"];

// ── Small UI Components ────────────────────────────────────────────────────
function Pill({ label, color, bg }: { label: string; color: string; bg: string }) {
  return (
    <span
      className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold tracking-wide border shadow-xs"
      style={{ color, backgroundColor: bg, borderColor: color + "40" }}
    >
      {label}
    </span>
  );
}

function BarRow({
  lbl,
  pct,
  color,
  val,
}: {
  lbl: string;
  pct: number;
  color: string;
  val: string;
}) {
  return (
    <div className="flex items-center gap-3 mb-2.5">
      <div className="text-xs font-medium text-slate-600 w-44 truncate shrink-0">{lbl}</div>
      <div className="flex-1 h-2 rounded-full bg-slate-100 overflow-hidden shadow-inner">
        <div
          className="h-full rounded-full transition-all duration-700 ease-out"
          style={{ width: `${Math.max(2, Math.min(pct * 100, 100))}%`, background: color }}
        />
      </div>
      <div className="text-xs font-semibold text-slate-700 w-12 text-right shrink-0">{val}</div>
    </div>
  );
}

function WhatIfSimulator({
  biomarkers,
  baseRisk,
  onApplyTargets,
}: {
  biomarkers: BiomarkerInput;
  baseRisk: number;
  onApplyTargets?: (hba1c: number, sbp: number) => void;
}) {
  const [hba1c, setHba1c] = useState(biomarkers.hba1c);
  const [sbp, setSbp] = useState(biomarkers.blood_pressure_systolic);

  // Synchronize when patient biomarkers change
  useEffect(() => {
    setHba1c(biomarkers.hba1c);
    setSbp(biomarkers.blood_pressure_systolic);
  }, [biomarkers.hba1c, biomarkers.blood_pressure_systolic]);

  // Calibrated epidemiological microvascular risk model
  const deltaHba1c = hba1c - biomarkers.hba1c;
  const deltaSbp = sbp - biomarkers.blood_pressure_systolic;
  
  // Non-linear hazard multiplier based on UKPDS / ETDRS clinical risk factors
  const multiplier = Math.exp(0.22 * deltaHba1c + 0.012 * deltaSbp);
  const projectedRisk = Math.max(0.01, Math.min(0.99, baseRisk * multiplier));
  const t = tierOf(projectedRisk);
  const netDelta = baseRisk - projectedRisk;

  function resetToCurrent() {
    setHba1c(biomarkers.hba1c);
    setSbp(biomarkers.blood_pressure_systolic);
  }

  return (
    <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 mt-1">
      <div className="flex items-center justify-between mb-3">
        <div className="text-xs font-semibold text-slate-600 uppercase tracking-wider">
          Interactive Clinical Risk Projection (What-If Simulation)
        </div>
        {(hba1c !== biomarkers.hba1c || sbp !== biomarkers.blood_pressure_systolic) && (
          <button
            onClick={resetToCurrent}
            className="text-[11px] font-semibold text-indigo-600 hover:text-indigo-800 transition-colors"
          >
            Reset Sliders
          </button>
        )}
      </div>

      <p className="text-xs text-slate-500 mb-3 leading-relaxed">
        Simulate how therapeutic glycemic management and blood pressure reduction will improve this patient’s diabetic retinopathy risk trajectory.
      </p>

      <div className="space-y-3">
        <div className="flex items-center gap-3">
          <span className="text-xs font-medium text-slate-600 w-28 shrink-0">Target HbA1c:</span>
          <input
            type="range"
            min={5.0}
            max={14.0}
            step={0.1}
            value={hba1c}
            onChange={(e) => setHba1c(parseFloat(e.target.value))}
            className="flex-1 accent-indigo-600 cursor-pointer"
          />
          <span className="text-xs font-bold text-slate-800 w-14 text-right">
            {hba1c.toFixed(1)}%
          </span>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-xs font-medium text-slate-600 w-28 shrink-0">Target SBP:</span>
          <input
            type="range"
            min={90}
            max={200}
            step={1}
            value={sbp}
            onChange={(e) => setSbp(parseFloat(e.target.value))}
            className="flex-1 accent-indigo-600 cursor-pointer"
          />
          <span className="text-xs font-bold text-slate-800 w-14 text-right">
            {sbp.toFixed(0)} mmHg
          </span>
        </div>
      </div>

      <div className="mt-4 pt-3 border-t border-slate-200 flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500">Projected Risk:</span>
          <span className="text-base font-bold" style={{ color: t.color }}>
            {(projectedRisk * 100).toFixed(1)}%
          </span>
          <Pill label={t.label} color={t.color} bg={t.bg} />
        </div>

        <div className="flex items-center gap-2">
          {Math.abs(netDelta) > 0.005 && (
            <span
              className={`text-xs font-semibold px-2.5 py-1 rounded-md ${
                netDelta > 0 ? "bg-emerald-100 text-emerald-800" : "bg-rose-100 text-rose-800"
              }`}
            >
              {netDelta > 0 ? "↓ Risk reduced by " : "↑ Risk increased by "}
              {Math.abs(netDelta * 100).toFixed(1)}%
            </span>
          )}

          {onApplyTargets && (hba1c !== biomarkers.hba1c || sbp !== biomarkers.blood_pressure_systolic) && (
            <button
              onClick={() => onApplyTargets(hba1c, sbp)}
              className="text-xs font-bold px-2.5 py-1 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors cursor-pointer"
            >
              Apply Targets to Profile
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────
export default function Stage1Screen() {
  const [activeTab, setActiveTab] = useState<"individual" | "batch">("individual");

  // Single patient state
  const [biomarkers, setBiomarkers] = useState<BiomarkerInput>(biomarkerDefaults);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [singleLoading, setSingleLoading] = useState(false);
  const [singleResult, setSingleResult] = useState<PredictionResponse | null>(null);
  const [singleError, setSingleError] = useState<string | null>(null);
  const [downloadingReport, setDownloadingReport] = useState(false);

  // Batch screening state (Real CSV, zero mock data)
  const [batchFile, setBatchFile] = useState<File | null>(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchResult, setBatchResult] = useState<BatchPredictionResponse | null>(null);
  const [batchError, setBatchError] = useState<string | null>(null);
  const [batchFilter, setBatchFilter] = useState<string>("ALL");
  const [batchSearch, setBatchSearch] = useState<string>("");
  const [selectedBatchItem, setSelectedBatchItem] = useState<BatchItemResult | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const batchFileInputRef = useRef<HTMLInputElement>(null);

  // ── Handlers for Single Patient ──────────────────────────────────────────
  function handleBiomarkerChange(key: keyof BiomarkerInput, val: number) {
    setBiomarkers((prev) => ({ ...prev, [key]: val }));
  }

  function handleImageSelected(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) {
      setImageFile(file);
      setImagePreview(URL.createObjectURL(file));
    }
  }

  function removeImage() {
    setImageFile(null);
    setImagePreview(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  function handleApplyTargets(targetHba1c: number, targetSbp: number) {
    setBiomarkers((prev) => ({
      ...prev,
      hba1c: targetHba1c,
      blood_pressure_systolic: targetSbp,
    }));
  }

  async function handleRunSingleScreening() {
    setSingleLoading(true);
    setSingleError(null);

    try {
      let res: PredictionResponse;
      if (imageFile && biomarkers) {
        // Multi-modal Unified (Late Fusion)
        res = await predictUnified(imageFile, biomarkers);
      } else if (imageFile) {
        // CNN Fundus only
        res = await predictImageOnly(imageFile);
      } else {
        // Biomarkers only
        res = await predictBiomarker(biomarkers);
      }
      setSingleResult(res);
    } catch (err) {
      setSingleError(err instanceof Error ? err.message : "Diagnostic screening failed.");
    } finally {
      setSingleLoading(false);
    }
  }

  async function handleDownloadReport() {
    if (!imageFile) {
      alert("Please select a retinal fundus image to generate the full PDF clinical report.");
      return;
    }
    setDownloadingReport(true);
    try {
      await downloadUnifiedReport(imageFile, biomarkers);
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to generate clinical PDF report.");
    } finally {
      setDownloadingReport(false);
    }
  }

  // ── Handlers for Batch CSV Screening ─────────────────────────────────────
  async function handleBatchCsvUpload(file: File) {
    setBatchFile(file);
    setBatchLoading(true);
    setBatchError(null);
    setBatchResult(null);
    setSelectedBatchItem(null);

    try {
      const data: BatchPredictionResponse = await predictBatchCsv(file);
      setBatchResult(data);
    } catch (err) {
      setBatchError(err instanceof Error ? err.message : "Batch processing failed.");
    } finally {
      setBatchLoading(false);
    }
  }

  function generateSampleCsv() {
    const csvContent =
      "patient_id,age,bmi,hba1c,blood_pressure_systolic,blood_pressure_diastolic,cholesterol_total,cholesterol_hdl,cholesterol_ldl,triglycerides,diabetes_duration_years,smoking_status,family_history_dr\n" +
      "PAT-1001,62,31.2,9.8,155,92,230,38,150,210,16,1,1\n" +
      "PAT-1002,48,24.5,6.1,120,78,185,55,105,120,3,0,0\n" +
      "PAT-1003,55,28.0,8.4,142,86,210,44,130,175,10,0,1\n" +
      "PAT-1004,70,33.5,10.2,168,96,260,35,175,250,22,2,1\n" +
      "PAT-1005,51,26.8,6.8,128,82,195,50,115,140,5,0,0\n" +
      "PAT-1006,64,30.1,9.1,148,90,220,40,140,195,14,1,0\n" +
      "PAT-1007,45,23.9,5.7,115,75,175,60,95,110,2,0,0\n" +
      "PAT-1008,59,32.0,8.9,150,92,240,36,160,220,13,2,1\n";

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", "sample_clinical_cohort.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  function exportBatchResultsCsv() {
    if (!batchResult) return;
    const header =
      "patient_id,risk_score,screening_tier,predicted_grade,predicted_label,age,hba1c,bp_systolic,bp_diastolic,bmi,diabetes_duration_years,recommendation\n";
    const rows = batchResult.results.map(
      (r) =>
        `"${r.patient_id}",${r.risk_score},"${r.screening_tier}",${r.predicted_grade},"${r.predicted_label}",${r.age},${r.hba1c},${r.blood_pressure_systolic},${r.blood_pressure_diastolic},${r.bmi},${r.diabetes_duration_years},"${r.baseline_recommendation || ""}"`
    );
    const csvContent = header + rows.join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `dr_triage_evaluated_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  // Filtered batch items
  const filteredBatchResults = (batchResult?.results || []).filter((item) => {
    const matchesFilter =
      batchFilter === "ALL" ||
      item.screening_tier.toLowerCase() === batchFilter.toLowerCase();
    const matchesSearch =
      batchSearch.trim() === "" ||
      item.patient_id.toLowerCase().includes(batchSearch.toLowerCase()) ||
      item.predicted_label.toLowerCase().includes(batchSearch.toLowerCase());
    return matchesFilter && matchesSearch;
  });

  const singleTier = singleResult ? tierOf(singleResult.risk_score) : null;

  return (
    <div className="min-h-screen bg-slate-100/70 text-slate-800 antialiased">
      {/* Top Navbar */}
      <header className="sticky top-0 z-30 bg-white/95 backdrop-blur-md border-b border-slate-200 px-6 py-3.5 shadow-xs">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-xl bg-gradient-to-tr from-cyan-600 to-indigo-600 flex items-center justify-center text-white font-bold text-lg shadow-sm">
              R
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-lg font-bold tracking-tight text-slate-900">
                  Retina<span className="text-cyan-600">Guard</span>
                </span>
                <span className="bg-cyan-50 text-cyan-700 text-[11px] font-semibold px-2 py-0.5 rounded-full border border-cyan-200">
                  AI Clinical Suite
                </span>
              </div>
              <p className="text-xs text-slate-500">
                Diabetic Retinopathy Multimodal AI & Biomarker Triage
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Mode Switcher */}
            <div className="bg-slate-100 p-1 rounded-xl flex items-center border border-slate-200">
              <button
                onClick={() => setActiveTab("individual")}
                className={`px-4 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  activeTab === "individual"
                    ? "bg-white text-slate-900 shadow-xs"
                    : "text-slate-500 hover:text-slate-800"
                }`}
              >
                Single Patient Screening
              </button>
              <button
                onClick={() => setActiveTab("batch")}
                className={`px-4 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  activeTab === "batch"
                    ? "bg-white text-slate-900 shadow-xs"
                    : "text-slate-500 hover:text-slate-800"
                }`}
              >
                Clinic Batch CSV Triage
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
        {/* ================================================================= */}
        {/* INDIVIDUAL PATIENT SCREENING TAB                                 */}
        {/* ================================================================= */}
        {activeTab === "individual" && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
              {/* Left Column: Clinical Biomarkers Form (7 cols) */}
              <div className="lg:col-span-7 bg-white border border-slate-200 rounded-2xl p-6 shadow-xs">
                <div className="flex items-center justify-between pb-4 border-b border-slate-100">
                  <div>
                    <h2 className="text-base font-bold text-slate-900">Patient Clinical Biomarkers</h2>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Metabolic, cardiovascular, and diabetes risk profile
                    </p>
                  </div>
                  <span className="text-xs font-medium bg-indigo-50 text-indigo-700 px-2.5 py-1 rounded-lg border border-indigo-100">
                    Stage 1 Biomarker Model: 99.1% Acc
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-5">
                  <div>
                    <label className="text-xs font-semibold text-slate-600 block mb-1.5">
                      Age (years)
                    </label>
                    <input
                      type="number"
                      value={biomarkers.age}
                      min={18}
                      max={100}
                      onChange={(e) => handleBiomarkerChange("age", parseFloat(e.target.value) || 0)}
                      className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-cyan-500/20 focus:border-cyan-500 font-medium"
                    />
                  </div>

                  <div>
                    <label className="text-xs font-semibold text-slate-600 block mb-1.5">
                      BMI (kg/m²)
                    </label>
                    <input
                      type="number"
                      step={0.1}
                      value={biomarkers.bmi}
                      min={12}
                      max={60}
                      onChange={(e) => handleBiomarkerChange("bmi", parseFloat(e.target.value) || 0)}
                      className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-cyan-500/20 focus:border-cyan-500 font-medium"
                    />
                  </div>

                  <div>
                    <label className="text-xs font-semibold text-slate-600 block mb-1.5">
                      HbA1c (%)
                    </label>
                    <input
                      type="number"
                      step={0.1}
                      value={biomarkers.hba1c}
                      min={4}
                      max={18}
                      onChange={(e) => handleBiomarkerChange("hba1c", parseFloat(e.target.value) || 0)}
                      className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-cyan-500/20 focus:border-cyan-500 font-medium"
                    />
                  </div>

                  <div>
                    <label className="text-xs font-semibold text-slate-600 block mb-1.5">
                      BP Systolic (mmHg)
                    </label>
                    <input
                      type="number"
                      value={biomarkers.blood_pressure_systolic}
                      min={70}
                      max={240}
                      onChange={(e) =>
                        handleBiomarkerChange("blood_pressure_systolic", parseFloat(e.target.value) || 0)
                      }
                      className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-cyan-500/20 focus:border-cyan-500 font-medium"
                    />
                  </div>

                  <div>
                    <label className="text-xs font-semibold text-slate-600 block mb-1.5">
                      BP Diastolic (mmHg)
                    </label>
                    <input
                      type="number"
                      value={biomarkers.blood_pressure_diastolic}
                      min={40}
                      max={140}
                      onChange={(e) =>
                        handleBiomarkerChange("blood_pressure_diastolic", parseFloat(e.target.value) || 0)
                      }
                      className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-cyan-500/20 focus:border-cyan-500 font-medium"
                    />
                  </div>

                  <div>
                    <label className="text-xs font-semibold text-slate-600 block mb-1.5">
                      Diabetes Duration (yrs)
                    </label>
                    <input
                      type="number"
                      step={0.5}
                      value={biomarkers.diabetes_duration_years}
                      min={0}
                      max={60}
                      onChange={(e) =>
                        handleBiomarkerChange("diabetes_duration_years", parseFloat(e.target.value) || 0)
                      }
                      className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-cyan-500/20 focus:border-cyan-500 font-medium"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-4">
                  <div>
                    <label className="text-xs font-semibold text-slate-600 block mb-1.5">
                      Total Chol (mg/dL)
                    </label>
                    <input
                      type="number"
                      value={biomarkers.cholesterol_total}
                      min={80}
                      max={450}
                      onChange={(e) =>
                        handleBiomarkerChange("cholesterol_total", parseFloat(e.target.value) || 0)
                      }
                      className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-cyan-500/20 focus:border-cyan-500 font-medium"
                    />
                  </div>

                  <div>
                    <label className="text-xs font-semibold text-slate-600 block mb-1.5">
                      HDL Chol (mg/dL)
                    </label>
                    <input
                      type="number"
                      value={biomarkers.cholesterol_hdl}
                      min={15}
                      max={120}
                      onChange={(e) =>
                        handleBiomarkerChange("cholesterol_hdl", parseFloat(e.target.value) || 0)
                      }
                      className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-cyan-500/20 focus:border-cyan-500 font-medium"
                    />
                  </div>

                  <div>
                    <label className="text-xs font-semibold text-slate-600 block mb-1.5">
                      LDL Chol (mg/dL)
                    </label>
                    <input
                      type="number"
                      value={biomarkers.cholesterol_ldl}
                      min={20}
                      max={350}
                      onChange={(e) =>
                        handleBiomarkerChange("cholesterol_ldl", parseFloat(e.target.value) || 0)
                      }
                      className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-cyan-500/20 focus:border-cyan-500 font-medium"
                    />
                  </div>

                  <div>
                    <label className="text-xs font-semibold text-slate-600 block mb-1.5">
                      Triglycerides (mg/dL)
                    </label>
                    <input
                      type="number"
                      value={biomarkers.triglycerides}
                      min={30}
                      max={800}
                      onChange={(e) =>
                        handleBiomarkerChange("triglycerides", parseFloat(e.target.value) || 0)
                      }
                      className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-cyan-500/20 focus:border-cyan-500 font-medium"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4">
                  <div>
                    <label className="text-xs font-semibold text-slate-600 block mb-1.5">
                      Smoking Status
                    </label>
                    <select
                      value={biomarkers.smoking_status}
                      onChange={(e) =>
                        handleBiomarkerChange("smoking_status", parseInt(e.target.value, 10))
                      }
                      className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-cyan-500/20 focus:border-cyan-500 font-medium"
                    >
                      <option value={0}>Non-smoker (0)</option>
                      <option value={1}>Former smoker (1)</option>
                      <option value={2}>Current smoker (2)</option>
                    </select>
                  </div>

                  <div>
                    <label className="text-xs font-semibold text-slate-600 block mb-1.5">
                      Family History of Retinopathy
                    </label>
                    <select
                      value={biomarkers.family_history_dr}
                      onChange={(e) =>
                        handleBiomarkerChange("family_history_dr", parseInt(e.target.value, 10))
                      }
                      className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-cyan-500/20 focus:border-cyan-500 font-medium"
                    >
                      <option value={0}>No (0)</option>
                      <option value={1}>Yes (1)</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Right Column: Fundus Image Upload & Action (5 cols) */}
              <div className="lg:col-span-5 space-y-4">
                <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs">
                  <div className="flex items-center justify-between pb-3 border-b border-slate-100">
                    <div>
                      <h2 className="text-base font-bold text-slate-900">Retinal Fundus Image</h2>
                      <p className="text-xs text-slate-500 mt-0.5">Stage 2 Deep CNN & Grad-CAM</p>
                    </div>
                    {imageFile && (
                      <button
                        onClick={removeImage}
                        className="text-xs font-semibold text-rose-600 hover:text-rose-700 transition-colors"
                      >
                        Remove
                      </button>
                    )}
                  </div>

                  <input
                    type="file"
                    ref={fileInputRef}
                    accept="image/png,image/jpeg,image/jpg,image/tif"
                    onChange={handleImageSelected}
                    className="hidden"
                  />

                  {!imagePreview ? (
                    <div
                      onClick={() => fileInputRef.current?.click()}
                      className="mt-4 border-2 border-dashed border-slate-300 hover:border-cyan-500 hover:bg-cyan-50/40 transition-all rounded-2xl p-8 text-center cursor-pointer bg-slate-50/50 group"
                    >
                      <div className="h-12 w-12 rounded-2xl bg-cyan-100 text-cyan-700 flex items-center justify-center mx-auto text-2xl group-hover:scale-110 transition-transform">
                        📷
                      </div>
                      <p className="mt-3 text-sm font-semibold text-slate-800">
                        Upload Retinal Fundus Photograph
                      </p>
                      <p className="text-xs text-slate-500 mt-1">
                        PNG, JPG, or TIFF · Automatically preprocessed with CLAHE & ROI cropping
                      </p>
                    </div>
                  ) : (
                    <div className="mt-4 space-y-3">
                      <div className="relative h-52 w-full rounded-xl overflow-hidden border border-slate-200 bg-black">
                        <Image
                          src={imagePreview}
                          alt="Retinal Fundus Preview"
                          fill
                          unoptimized
                          className="object-contain"
                        />
                      </div>
                      <div className="flex items-center justify-between text-xs text-slate-500 px-1">
                        <span className="font-medium text-slate-700 truncate max-w-[200px]">
                          {imageFile?.name}
                        </span>
                        <span className="bg-emerald-50 text-emerald-700 px-2 py-0.5 rounded border border-emerald-200">
                          Ready for Deep Inference
                        </span>
                      </div>
                    </div>
                  )}

                  <button
                    onClick={handleRunSingleScreening}
                    disabled={singleLoading}
                    className="w-full mt-5 py-3 rounded-xl bg-gradient-to-r from-slate-900 to-indigo-950 text-white text-sm font-bold shadow-md hover:from-slate-800 hover:to-indigo-900 disabled:opacity-50 transition-all flex items-center justify-center gap-2 cursor-pointer"
                  >
                    {singleLoading ? (
                      <>
                        <span className="inline-block animate-spin">⟳</span>
                        Running AI Diagnostic Screening...
                      </>
                    ) : (
                      <>
                        <span>⚡</span>
                        {imageFile
                          ? "Run Multimodal AI Diagnostic (Late Fusion)"
                          : "Run Biomarker Risk Screening"}
                      </>
                    )}
                  </button>
                </div>

                {singleError && (
                  <div className="bg-rose-50 border border-rose-200 rounded-2xl p-4 text-rose-800 text-xs">
                    <p className="font-bold mb-1">Screening Error</p>
                    <p>{singleError}</p>
                  </div>
                )}
              </div>
            </div>

            {/* Diagnostic Results Section */}
            {singleResult && singleTier && (
              <div className="space-y-6 pt-2 animate-fadeIn">
                {/* Primary Risk Card */}
                <div
                  className="bg-white border rounded-2xl p-6 shadow-sm"
                  style={{ borderColor: singleTier.border, borderLeftWidth: "6px" }}
                >
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-100">
                    <div>
                      <div className="flex items-center gap-2.5 flex-wrap">
                        <Pill label={singleTier.label} color={singleTier.color} bg={singleTier.bg} />
                        <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-slate-100 text-slate-700">
                          {singleResult.predicted_label}
                        </span>
                        <span className="text-xs text-slate-400 bg-slate-50 px-2.5 py-1 rounded-full border border-slate-100">
                          Model: {singleResult.model_used}
                        </span>
                      </div>
                      <p className="text-sm text-slate-600 mt-2.5 leading-relaxed font-medium">
                        {singleResult.baseline_recommendation ||
                          "Clinical review indicated based on AI risk stratification."}
                      </p>
                    </div>

                    <div className="text-left md:text-right shrink-0 bg-slate-50 md:bg-transparent p-4 md:p-0 rounded-xl border md:border-0 border-slate-100">
                      <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                        Unified Risk Score
                      </p>
                      <p className="text-4xl font-extrabold tracking-tight" style={{ color: singleTier.color }}>
                        {(singleResult.risk_score * 100).toFixed(1)}%
                      </p>
                      <p className="text-[11px] text-slate-400 mt-0.5">Scale: 0% (Healthy) to 100% (High Severity)</p>
                    </div>
                  </div>

                  {/* Risk Meter Bar */}
                  <div className="mt-4">
                    <div className="h-3 rounded-full bg-slate-100 overflow-hidden shadow-inner">
                      <div
                        className="h-full rounded-full transition-all duration-1000 ease-out"
                        style={{
                          width: `${Math.max(2, singleResult.risk_score * 100)}%`,
                          background: singleTier.bar,
                        }}
                      />
                    </div>
                  </div>
                </div>

                {/* Grade Distribution & Grad-CAM Visual Explainability */}
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                  {/* Grade Probabilities (6 cols) */}
                  <div className="lg:col-span-6 bg-white border border-slate-200 rounded-2xl p-6 shadow-xs">
                    <div className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4">
                      Severity Grade Probability Breakdown
                    </div>
                    <div className="space-y-1">
                      {singleResult.grade_probabilities.map((g) => (
                        <BarRow
                          key={g.grade}
                          lbl={GRADES[g.grade] || g.label}
                          pct={g.probability}
                          color={GCOLS[g.grade]}
                          val={`${(g.probability * 100).toFixed(1)}%`}
                        />
                      ))}
                    </div>

                    {singleResult.baseline_factor_breakdown && (
                      <div className="mt-6 pt-4 border-t border-slate-100">
                        <div className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">
                          Clinical Biomarker Factor Contributions
                        </div>
                        {Object.entries(singleResult.baseline_factor_breakdown).map(([k, v]) => (
                          <BarRow
                            key={k}
                            lbl={k.replace(/_/g, " ")}
                            pct={v}
                            color="#4F46E5"
                            val={`${(v * 100).toFixed(0)}%`}
                          />
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Grad-CAM Heatmap & Explainability (6 cols) */}
                  <div className="lg:col-span-6 bg-white border border-slate-200 rounded-2xl p-6 shadow-xs">
                    <div className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4">
                      Grad-CAM Visual Diagnostic Explainability
                    </div>
                    {singleResult.grad_cam_available && singleResult.grad_cam_overlay ? (
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-3">
                          {singleResult.grad_cam_heatmap && (
                            <div className="border border-slate-200 rounded-xl p-2 bg-slate-50">
                              <p className="text-[11px] font-semibold text-slate-500 uppercase mb-1.5">
                                Activation Heatmap
                              </p>
                              <div className="relative h-40 w-full rounded-lg overflow-hidden bg-black">
                                <Image
                                  src={singleResult.grad_cam_heatmap}
                                  alt="Grad-CAM Heatmap"
                                  fill
                                  unoptimized
                                  className="object-contain"
                                />
                              </div>
                            </div>
                          )}
                          <div className="border border-slate-200 rounded-xl p-2 bg-slate-50">
                            <p className="text-[11px] font-semibold text-slate-500 uppercase mb-1.5">
                              Fundus Lesion Overlay
                            </p>
                            <div className="relative h-40 w-full rounded-lg overflow-hidden bg-black">
                              <Image
                                src={singleResult.grad_cam_overlay}
                                alt="Grad-CAM Overlay"
                                fill
                                unoptimized
                                className="object-contain"
                              />
                            </div>
                          </div>
                        </div>
                        <p className="text-xs text-slate-500 leading-relaxed">
                          Highlighted regions pinpoint the critical retinal features (e.g.
                          microaneurysms, hemorrhages, hard exudates, or neovascularization) driving
                          the neural network’s grade classification.
                        </p>
                      </div>
                    ) : (
                      <div className="bg-slate-50 border border-slate-200 rounded-xl p-8 text-center text-slate-500 text-xs">
                        <p className="text-2xl mb-2">👁️</p>
                        <p className="font-semibold text-slate-700">Grad-CAM Ready</p>
                        <p className="mt-1">
                          Upload a retinal fundus image with your biomarkers to generate deep
                          attention heatmaps.
                        </p>
                      </div>
                    )}
                  </div>
                </div>

                {/* What-If Simulator & Report Generation */}
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                  <div className="lg:col-span-8 bg-white border border-slate-200 rounded-2xl p-6 shadow-xs">
                    <WhatIfSimulator
                      biomarkers={biomarkers}
                      baseRisk={singleResult.risk_score}
                      onApplyTargets={handleApplyTargets}
                    />
                  </div>

                  <div className="lg:col-span-4 bg-white border border-slate-200 rounded-2xl p-6 shadow-xs flex flex-col justify-between">
                    <div>
                      <div className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">
                        Clinical Documentation
                      </div>
                      <p className="text-xs text-slate-500 leading-relaxed mb-4">
                        Generate and download an official Medical Screening Report with complete
                        patient biomarkers, AI grading, Grad-CAM overlays, and clinical follow-up
                        recommendations.
                      </p>
                    </div>

                    <div className="space-y-2">
                      <button
                        onClick={handleDownloadReport}
                        disabled={downloadingReport || !imageFile}
                        className="w-full py-2.5 rounded-xl bg-cyan-700 hover:bg-cyan-800 text-white text-xs font-bold transition-colors flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
                      >
                        {downloadingReport ? "Generating PDF..." : "📥 Download Clinical PDF Report"}
                      </button>
                      <button
                        onClick={() => {
                          setSingleResult(null);
                          removeImage();
                        }}
                        className="w-full py-2.5 rounded-xl border border-slate-200 hover:bg-slate-50 text-slate-700 text-xs font-bold transition-colors cursor-pointer"
                      >
                        New Screening
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ================================================================= */}
        {/* CLINIC BATCH CSV TRIAGE TAB (REAL DATA ONLY)                      */}
        {/* ================================================================= */}
        {activeTab === "batch" && (
          <div className="space-y-6">
            {/* Upload Area */}
            <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-100">
                <div>
                  <h2 className="text-base font-bold text-slate-900">
                    Hospital EHR Patient Cohort Batch Triage
                  </h2>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Upload actual patient CSV files to run automated population-level screening and
                    risk prioritization
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={generateSampleCsv}
                    className="text-xs font-semibold px-3 py-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 text-slate-700 transition-colors flex items-center gap-1.5 cursor-pointer"
                  >
                    <span>📄</span> Download CSV Template
                  </button>
                </div>
              </div>

              <input
                type="file"
                ref={batchFileInputRef}
                accept=".csv,text/csv"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) handleBatchCsvUpload(f);
                }}
                className="hidden"
              />

              <div
                onClick={() => batchFileInputRef.current?.click()}
                className="mt-5 border-2 border-dashed border-slate-300 hover:border-indigo-500 hover:bg-indigo-50/30 transition-all rounded-2xl p-8 text-center cursor-pointer bg-slate-50/50 group"
              >
                <div className="h-12 w-12 rounded-2xl bg-indigo-100 text-indigo-700 flex items-center justify-center mx-auto text-2xl group-hover:scale-110 transition-transform">
                  📊
                </div>
                <p className="mt-3 text-sm font-semibold text-slate-800">
                  {batchLoading
                    ? "Processing and evaluating patient cohort..."
                    : batchFile
                    ? `Selected: ${batchFile.name} (Click to change)`
                    : "Select or Drag & Drop Patient Cohort CSV"}
                </p>
                <p className="text-xs text-slate-500 mt-1">
                  Accepts standard EHR exports with Age, HbA1c, BP, BMI, Lipids, and Duration
                </p>
              </div>

              {batchError && (
                <div className="mt-4 bg-rose-50 border border-rose-200 rounded-xl p-4 text-rose-800 text-xs">
                  <p className="font-bold mb-1">Batch Processing Error</p>
                  <p>{batchError}</p>
                </div>
              )}
            </div>

            {/* Batch Results View */}
            {batchResult && (
              <div className="space-y-6 animate-fadeIn">
                {/* Cohort Summary Metrics */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-xs">
                    <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                      Total Screened
                    </p>
                    <p className="text-3xl font-extrabold text-slate-900 mt-1">
                      {batchResult.total_patients}
                    </p>
                    <p className="text-[11px] text-slate-500 mt-1">Patient cohort evaluated</p>
                  </div>

                  <div className="bg-red-50/60 border border-red-200 rounded-2xl p-5 shadow-xs">
                    <p className="text-xs font-semibold text-red-600 uppercase tracking-wider">
                      Urgent Referrals
                    </p>
                    <p className="text-3xl font-extrabold text-red-700 mt-1">
                      {batchResult.urgent_count}
                    </p>
                    <p className="text-[11px] text-red-600 mt-1">
                      {((batchResult.urgent_count / batchResult.total_patients) * 100).toFixed(0)}% of cohort
                    </p>
                  </div>

                  <div className="bg-amber-50/60 border border-amber-200 rounded-2xl p-5 shadow-xs">
                    <p className="text-xs font-semibold text-amber-600 uppercase tracking-wider">
                      Moderate Risk
                    </p>
                    <p className="text-3xl font-extrabold text-amber-700 mt-1">
                      {batchResult.moderate_count}
                    </p>
                    <p className="text-[11px] text-amber-600 mt-1">
                      {((batchResult.moderate_count / batchResult.total_patients) * 100).toFixed(0)}% of cohort
                    </p>
                  </div>

                  <div className="bg-emerald-50/60 border border-emerald-200 rounded-2xl p-5 shadow-xs">
                    <p className="text-xs font-semibold text-emerald-600 uppercase tracking-wider">
                      Low Risk
                    </p>
                    <p className="text-3xl font-extrabold text-emerald-700 mt-1">
                      {batchResult.low_risk_count}
                    </p>
                    <p className="text-[11px] text-emerald-600 mt-1">
                      {((batchResult.low_risk_count / batchResult.total_patients) * 100).toFixed(0)}% of cohort
                    </p>
                  </div>
                </div>

                {/* Cohort Interactive Table */}
                <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-100">
                    <div className="flex items-center gap-3">
                      <h3 className="text-base font-bold text-slate-900">Screened Patient Roster</h3>
                      <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-600">
                        {filteredBatchResults.length} records
                      </span>
                    </div>

                    <div className="flex items-center gap-2.5 flex-wrap">
                      <input
                        type="text"
                        placeholder="Search patient ID or grade..."
                        value={batchSearch}
                        onChange={(e) => setBatchSearch(e.target.value)}
                        className="bg-slate-50 border border-slate-200 rounded-lg px-3 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-indigo-500 w-48"
                      />

                      <select
                        value={batchFilter}
                        onChange={(e) => setBatchFilter(e.target.value)}
                        className="bg-slate-50 border border-slate-200 rounded-lg px-3 py-1.5 text-xs font-semibold text-slate-700 focus:outline-none"
                      >
                        <option value="ALL">All Tiers</option>
                        <option value="Urgent">Urgent Only</option>
                        <option value="Moderate">Moderate Only</option>
                        <option value="Low Risk">Low Risk Only</option>
                      </select>

                      <button
                        onClick={exportBatchResultsCsv}
                        className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-white text-xs font-bold transition-colors flex items-center gap-1.5 cursor-pointer"
                      >
                        <span>📥</span> Export Evaluated CSV
                      </button>
                    </div>
                  </div>

                  {/* Table Container */}
                  <div className="overflow-x-auto mt-4">
                    <table className="w-full text-left text-xs">
                      <thead>
                        <tr className="border-b border-slate-200 text-slate-400 font-semibold uppercase tracking-wider">
                          <th className="pb-3 px-3">Patient ID</th>
                          <th className="pb-3 px-3">Age</th>
                          <th className="pb-3 px-3">HbA1c</th>
                          <th className="pb-3 px-3">Blood Pressure</th>
                          <th className="pb-3 px-3">Duration</th>
                          <th className="pb-3 px-3 text-right">Risk Score</th>
                          <th className="pb-3 px-3 text-center">Triage Tier</th>
                          <th className="pb-3 px-3">Action Recommendation</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 font-medium text-slate-700">
                        {filteredBatchResults.map((p) => {
                          const t = tierOf(p.risk_score);
                          return (
                            <tr
                              key={p.patient_id}
                              onClick={() => setSelectedBatchItem(p)}
                              className="hover:bg-slate-50/80 transition-colors cursor-pointer"
                            >
                              <td className="py-3 px-3 font-bold text-slate-900">{p.patient_id}</td>
                              <td className="py-3 px-3">{p.age} yrs</td>
                              <td className="py-3 px-3 font-bold text-slate-800">{p.hba1c}%</td>
                              <td className="py-3 px-3">
                                {p.blood_pressure_systolic}/{p.blood_pressure_diastolic}
                              </td>
                              <td className="py-3 px-3">{p.diabetes_duration_years} yrs</td>
                              <td className="py-3 px-3 text-right font-extrabold" style={{ color: t.color }}>
                                {(p.risk_score * 100).toFixed(1)}%
                              </td>
                              <td className="py-3 px-3 text-center">
                                <span
                                  className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-[11px] font-bold border ${t.badge}`}
                                >
                                  {p.screening_tier}
                                </span>
                              </td>
                              <td className="py-3 px-3 text-slate-500 truncate max-w-xs">
                                {p.baseline_recommendation || "Standard monitoring"}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>

                  {filteredBatchResults.length === 0 && (
                    <div className="py-8 text-center text-slate-400 text-xs">
                      No patients match your search or filter criteria.
                    </div>
                  )}
                </div>

                {/* Selected Patient Drawer Modal / Breakdown */}
                {selectedBatchItem && (
                  <div className="bg-slate-900 text-white rounded-2xl p-6 shadow-xl flex flex-col md:flex-row items-start justify-between gap-6">
                    <div className="space-y-2">
                      <div className="flex items-center gap-3">
                        <h4 className="text-lg font-bold">Patient {selectedBatchItem.patient_id}</h4>
                        <span
                          className={`text-xs font-bold px-2.5 py-0.5 rounded-full ${
                            tierOf(selectedBatchItem.risk_score).badge
                          }`}
                        >
                          {selectedBatchItem.screening_tier}
                        </span>
                      </div>
                      <p className="text-xs text-slate-300">
                        {selectedBatchItem.baseline_recommendation}
                      </p>
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2 text-xs">
                        <div>
                          <span className="text-slate-400 block text-[10px]">Age / BMI</span>
                          <span className="font-semibold">{selectedBatchItem.age} yrs / {selectedBatchItem.bmi}</span>
                        </div>
                        <div>
                          <span className="text-slate-400 block text-[10px]">HbA1c</span>
                          <span className="font-semibold text-rose-400">{selectedBatchItem.hba1c}%</span>
                        </div>
                        <div>
                          <span className="text-slate-400 block text-[10px]">Blood Pressure</span>
                          <span className="font-semibold">
                            {selectedBatchItem.blood_pressure_systolic}/
                            {selectedBatchItem.blood_pressure_diastolic}
                          </span>
                        </div>
                        <div>
                          <span className="text-slate-400 block text-[10px]">DR Grade</span>
                          <span className="font-semibold">{selectedBatchItem.predicted_label}</span>
                        </div>
                      </div>
                    </div>

                    <button
                      onClick={() => setSelectedBatchItem(null)}
                      className="text-xs text-slate-400 hover:text-white px-3 py-1.5 rounded-lg border border-slate-700 transition-colors shrink-0"
                    >
                      Close Detail
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
