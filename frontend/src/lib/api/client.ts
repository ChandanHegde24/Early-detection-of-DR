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
      // Keep fallback message when API does not return JSON
    }
    throw new Error(message);
  }

  return (await res.json()) as PredictionResponse;
}
