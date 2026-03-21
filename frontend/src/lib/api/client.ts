import type { BiomarkerInput, PredictionResponse } from "@/lib/api/types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export async function healthCheck() {
  const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}

export async function predictUnified(
  imageFile: File,
  biomarkers: BiomarkerInput,
): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("file", imageFile);

  for (const [key, value] of Object.entries(biomarkers)) {
    form.append(key, String(value));
  }

  const res = await fetch(`${API_BASE}/predict/unified`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    let message = "Unified prediction failed";
    try {
      const payload = await res.json();
      if (payload?.detail) {
        message = String(payload.detail);
      }
    } catch {
    }
    throw new Error(message);
  }

  return (await res.json()) as PredictionResponse;
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

  const res = await fetch(`${API_BASE}/predict/unified/report`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    let message = "Failed to generate PDF report";
    try {
      const payload = await res.json();
      if (payload?.detail) {
        message = String(payload.detail);
      }
    } catch {
    }
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
