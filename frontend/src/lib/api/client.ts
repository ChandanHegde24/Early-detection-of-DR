import type { BiomarkerInput, PredictionResponse } from "@/lib/api/types";

export function getApiBase(): string {
  if (process.env.NEXT_PUBLIC_API_BASE_URL) {
    return process.env.NEXT_PUBLIC_API_BASE_URL;
  }
  if (typeof window !== "undefined" && window.location.hostname) {
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }
  return "http://localhost:8000";
}

async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  const base = getApiBase();
  const url = `${base}${path}`;
  try {
    return await fetch(url, init);
  } catch (error) {
    if (error instanceof TypeError && error.message.toLowerCase().includes("fetch")) {
      throw new Error(
        `Unable to reach backend API at ${base}. Please ensure the backend server is running (uvicorn api.main:app --reload --port 8000).`
      );
    }
    throw error;
  }
}

export async function healthCheck() {
  const res = await apiFetch(`/health`, { cache: "no-store" });
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}

/** Stage 1 — biomarkers only, no image required */
export async function predictBiomarker(
  biomarkers: BiomarkerInput,
): Promise<PredictionResponse> {
  const res = await apiFetch(`/predict/biomarker`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ biomarkers }),
  });

  if (!res.ok) {
    let message = "Biomarker prediction failed";
    try {
      const payload = await res.json();
      if (payload?.detail) message = String(payload.detail);
    } catch {}
    throw new Error(message);
  }

  return (await res.json()) as PredictionResponse;
}

/** Stage 2 / Unified — biomarkers + retinal image */
export async function predictUnified(
  imageFile: File,
  biomarkers: BiomarkerInput,
): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("file", imageFile);
  for (const [key, value] of Object.entries(biomarkers)) {
    form.append(key, String(value));
  }

  const res = await apiFetch(`/predict/unified`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    let message = "Unified prediction failed";
    try {
      const payload = await res.json();
      if (payload?.detail) message = String(payload.detail);
    } catch {}
    throw new Error(message);
  }

  return (await res.json()) as PredictionResponse;
}

export async function predictImageOnly(imageFile: File): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("file", imageFile);

  const res = await apiFetch(`/predict/image`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    let message = "Image prediction failed";
    try {
      const payload = await res.json();
      if (payload?.detail) message = String(payload.detail);
    } catch {}
    throw new Error(message);
  }

  return (await res.json()) as PredictionResponse;
}

/** Batch Screening — upload real patient CSV */
export async function predictBatchCsv(csvFile: File) {
  const form = new FormData();
  form.append("file", csvFile);

  const res = await apiFetch(`/predict/batch`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    let message = "Batch processing failed";
    try {
      const payload = await res.json();
      if (payload?.detail) message = String(payload.detail);
    } catch {}
    throw new Error(message);
  }

  return (await res.json());
}

export async function downloadUnifiedReport(
  imageFile: File,
  biomarkers: BiomarkerInput,
): Promise<void> {
  const form = new FormData();
  form.append("file", imageFile);
  for (const [key, value] of Object.entries(biomarkers)) {
    form.append(key, String(value));
  }

  const res = await apiFetch(`/predict/unified/report`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    let message = "Failed to generate PDF report";
    try {
      const payload = await res.json();
      if (payload?.detail) message = String(payload.detail);
    } catch {}
    throw new Error(message);
  }

  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "dr_clinical_report.pdf";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.URL.revokeObjectURL(url);
}

