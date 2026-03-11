"use client";

import { DemoUploader, type DemoSample } from "@/components/DemoUploader";
import { ResultViewer } from "@/components/ResultViewer";
import { useState } from "react";

export default function DemoPage() {
  const [activeSample, setActiveSample] = useState<DemoSample | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleProcessingStart = (sample: DemoSample | null) => {
    setIsProcessing(true);
    setActiveSample(sample);

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
          This page simulates the VisionFlow pipeline on a set of curated sample
          frames that mirror real traffic scenarios processed by the Python app.
        </p>
      </div>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1.4fr)]">
        <DemoUploader onProcessingStart={handleProcessingStart} />
        <ResultViewer
          activeSample={activeSample}
          inputUrl={activeSample?.inputPath ?? null}
          isProcessing={isProcessing}
        />
      </div>
    </div>
  );
}

