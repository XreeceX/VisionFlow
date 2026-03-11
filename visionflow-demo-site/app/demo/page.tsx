"use client";

import { DemoUploader, type DemoSample } from "@/components/DemoUploader";
import { ResultViewer } from "@/components/ResultViewer";
import { useState } from "react";

export default function DemoPage() {
  const [activeSample, setActiveSample] = useState<DemoSample | null>(null);
  const [inputUrl, setInputUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleProcessingStart = (sample: DemoSample | null, fileUrl?: string) => {
    setIsProcessing(true);
    setActiveSample(sample);
    setInputUrl(fileUrl ?? sample?.inputPath ?? null);

    // Simulate model inference time
    setTimeout(() => {
      setIsProcessing(false);
    }, 2200);
  };

  return (
    <div className="py-8 space-y-6">
      <div className="max-w-xl">
        <h1 className="text-2xl font-semibold text-slate-50 sm:text-3xl">
          Interactive demo
        </h1>
        <p className="mt-2 text-sm text-slate-400">
          This page simulates the VisionFlow pipeline on curated sample frames
          or an image you upload. Processing is visual-only and happens entirely
          in your browser.
        </p>
      </div>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1.4fr)]">
        <DemoUploader onProcessingStart={handleProcessingStart} />
        <ResultViewer
          activeSample={activeSample}
          inputUrl={inputUrl}
          isProcessing={isProcessing}
        />
      </div>
    </div>
  );
}

