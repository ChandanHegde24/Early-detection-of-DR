import { useState } from "react";
import { predictUnified } from "@/lib/api/client";
import type { BiomarkerInput, PredictionResponse } from "@/lib/api/types";

export function usePrediction() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function retryWithBackoff<T>(
    fn: () => Promise<T>,
    maxRetries = 3,
    delayMs = 1000
  ): Promise<T> {
    for (let i = 0; i < maxRetries; i++) {
      try {
        return await fn();
      } catch (err) {
        if (i === maxRetries - 1) throw err;
        await new Promise(resolve => setTimeout(resolve, delayMs * Math.pow(2, i)));
      }
    }
    throw new Error("Max retries exceeded");
  }

  async function submitUnified(imageFile: File, biomarkers: BiomarkerInput) {
    setLoading(true);
    setError(null);
    try {
      const response = await retryWithBackoff(() => predictUnified(imageFile, biomarkers));
      setResult(response);
      return response;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to run prediction";
      setError(message);
      setResult(null);
      throw err;
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setResult(null);
    setError(null);
  }

  return {
    loading,
    result,
    error,
    submitUnified,
    reset,
  };
}
