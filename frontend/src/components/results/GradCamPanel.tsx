import type { PredictionResponse } from "@/lib/api/types";
import Image from "next/image";

type GradCamPanelProps = {
  result: PredictionResponse;
};

export function GradCamPanel({ result }: GradCamPanelProps) {
  if (!result.grad_cam_available) {
    return (
      <section className="rounded-2xl border border-slate-200 bg-white/90 p-5 shadow-sm">
        <h3 className="text-base font-semibold text-slate-900">Grad-CAM</h3>
        <p className="mt-2 text-sm text-slate-600">Grad-CAM is not available for this prediction.</p>
      </section>
    );
  }

  return (
    <section className="rounded-2xl border border-cyan-100 bg-white/90 p-5 shadow-sm">
      <h3 className="text-base font-semibold text-slate-900">Grad-CAM Interpretability</h3>
      <p className="mt-1 text-sm text-slate-600">
        Hotter regions indicate retinal areas that most influenced the model decision.
      </p>

      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ImageCard title="Heatmap" src={result.grad_cam_heatmap} />
        <ImageCard title="Overlay on Fundus" src={result.grad_cam_overlay} />
      </div>
    </section>
  );
}

function ImageCard({ title, src }: { title: string; src: string | null }) {
  return (
    <div className="rounded-xl border border-slate-200 p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-600">{title}</p>
      {src ? (
        <Image
          src={src}
          alt={title}
          width={720}
          height={420}
          unoptimized
          className="mt-2 h-56 w-full rounded-lg border border-slate-100 object-cover"
        />
      ) : (
        <p className="mt-3 text-sm text-slate-500">No image returned by API.</p>
      )}
    </div>
  );
}
