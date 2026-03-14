import { useEffect, useState } from "react";
import Image from "next/image";

type FundusUploaderProps = {
  file: File | null;
  onFileChange: (file: File | null) => void;
  disabled?: boolean;
};

export function FundusUploader({ file, onFileChange, disabled = false }: FundusUploaderProps) {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    return () => {
      URL.revokeObjectURL(url);
    };
  }, [file]);

  return (
    <section className="rounded-2xl border border-cyan-100 bg-white/90 p-5 shadow-sm">
      <h2 className="text-lg font-semibold text-slate-900">Fundus Image</h2>
      <p className="mt-1 text-sm text-slate-600">
        Upload a retinal fundus photograph in PNG or JPEG format.
      </p>

      <label className="mt-4 flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-cyan-300 bg-cyan-50/60 p-6 text-center transition hover:border-cyan-500">
        <span className="text-sm font-medium text-cyan-900">Choose image</span>
        <span className="mt-1 text-xs text-cyan-700">Recommended: centered, high-contrast retina</span>
        <input
          className="hidden"
          type="file"
          accept="image/png,image/jpeg"
          disabled={disabled}
          onChange={(event) => onFileChange(event.target.files?.[0] ?? null)}
        />
      </label>

      {file ? (
        <div className="mt-4 space-y-3">
          <p className="text-xs text-slate-600">Selected: {file.name}</p>
          {previewUrl ? (
            <Image
              src={previewUrl}
              alt="Fundus preview"
              width={800}
              height={420}
              unoptimized
              className="h-56 w-full rounded-xl border border-slate-200 object-cover"
            />
          ) : null}
          <button
            type="button"
            disabled={disabled}
            onClick={() => onFileChange(null)}
            className="rounded-lg border border-slate-300 px-3 py-2 text-xs font-medium text-slate-700 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-60"
          >
            Remove image
          </button>
        </div>
      ) : null}
    </section>
  );
}
