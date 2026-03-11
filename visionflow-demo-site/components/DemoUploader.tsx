"use client";

import { useState, ChangeEvent } from "react";

export type DemoSample = {
  id: string;
  label: string;
  inputPath: string;
  resultPath: string;
  description: string;
};

const SAMPLE_IMAGES: DemoSample[] = [
  {
    id: "intersection-1",
    label: "Busy 4-way intersection",
    inputPath: "/sample-images/intersection-1.svg",
    resultPath: "/results/intersection-1-result.svg",
    description:
      "Multiple vehicles detected with per-lane counts and annotated license plates."
  },
  {
    id: "emergency-vehicle",
    label: "Emergency vehicle priority",
    inputPath: "/sample-images/emergency-vehicle.svg",
    resultPath: "/results/emergency-vehicle-result.svg",
    description:
      "Emergency vehicle plate detected and prioritized with green-wave lane assignment."
  }
];

type DemoUploaderProps = {
  onProcessingStart: (sample: DemoSample | null, fileUrl?: string) => void;
};

export function DemoUploader({ onProcessingStart }: DemoUploaderProps) {
  const [selectedId, setSelectedId] = useState<string | null>(
    SAMPLE_IMAGES[0]?.id ?? null
  );
  const [isUploading, setIsUploading] = useState(false);

  const handleSampleRun = () => {
    const sample = SAMPLE_IMAGES.find((s) => s.id === selectedId) ?? null;
    onProcessingStart(sample ?? null);
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const objectUrl = URL.createObjectURL(file);
    setIsUploading(true);
    onProcessingStart(null, objectUrl);
    setTimeout(() => setIsUploading(false), 300);
  };

  return (
    <div className="space-y-5 glass-panel rounded-2xl p-5 sm:p-6">
      <div>
        <h1 className="text-lg font-semibold text-slate-50 sm:text-xl">
          Interactive demo
        </h1>
        <p className="mt-1 text-xs text-slate-400">
          Upload a traffic frame or use a curated sample, then simulate the
          VisionFlow processing pipeline.
        </p>
      </div>

      <div className="space-y-3">
        <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
          1. Choose a sample
        </p>
        <div className="grid gap-3 sm:grid-cols-2">
          {SAMPLE_IMAGES.map((sample) => (
            <button
              key={sample.id}
              type="button"
              onClick={() => setSelectedId(sample.id)}
              className={`rounded-xl border px-3 py-3 text-left text-xs transition hover:border-accent/70 hover:bg-slate-900/60 ${
                selectedId === sample.id
                  ? "border-accent bg-slate-900/80 shadow-glow"
                  : "border-slate-800 bg-slate-950/60"
              }`}
            >
              <p className="font-medium text-slate-100">{sample.label}</p>
              <p className="mt-1 text-[0.7rem] text-slate-400 line-clamp-2">
                {sample.description}
              </p>
            </button>
          ))}
        </div>
        <button
          type="button"
          onClick={handleSampleRun}
          className="inline-flex items-center justify-center rounded-full bg-gradient-to-r from-accent via-indigo-500 to-sky-500 px-4 py-2 text-xs font-medium text-white shadow-glow transition hover:brightness-110"
        >
          Run pipeline on sample
        </button>
      </div>

      <div className="space-y-3">
        <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
          2. Or upload your own frame
        </p>
        <label className="flex cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-slate-700 bg-slate-950/60 px-4 py-6 text-center text-xs text-slate-400 transition hover:border-accent/80 hover:bg-slate-900/70">
          <span className="mb-2 rounded-full bg-slate-900 px-3 py-1 text-[0.65rem] font-medium text-slate-300">
            Drop a .jpg / .png frame
          </span>
          <span className="text-[0.7rem]">
            The image is processed only in your browser. The demo simulates
            VisionFlow behaviour visually and does not send files to a server.
          </span>
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
          />
        </label>
        {isUploading && (
          <p className="text-[0.7rem] text-emerald-300">
            Upload received – simulating processing…
          </p>
        )}
      </div>
    </div>
  );
}

