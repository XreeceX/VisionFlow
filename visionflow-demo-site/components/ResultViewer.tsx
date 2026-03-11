"use client";

import Image from "next/image";
import { useEffect, useState } from "react";
import type { DemoSample } from "./DemoUploader";

type ResultViewerProps = {
  activeSample: DemoSample | null;
  inputUrl: string | null;
  isProcessing: boolean;
};

export function ResultViewer({
  activeSample,
  inputUrl,
  isProcessing
}: ResultViewerProps) {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (!isProcessing) {
      setProgress(100);
      return;
    }

    setProgress(0);
    const start = Date.now();
    const duration = 2200;

    const interval = setInterval(() => {
      const elapsed = Date.now() - start;
      const nextProgress = Math.min(100, (elapsed / duration) * 100);
      setProgress(nextProgress);

      if (nextProgress >= 100) {
        clearInterval(interval);
      }
    }, 120);

    return () => clearInterval(interval);
  }, [isProcessing, activeSample, inputUrl]);

  const showResult = !isProcessing && (activeSample || inputUrl);

  return (
    <div className="glass-panel rounded-2xl p-5 sm:p-6 space-y-4">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-sm font-semibold text-slate-50 sm:text-base">
          Input vs. VisionFlow output
        </h2>
        <span className="rounded-full border border-slate-700 bg-slate-900/80 px-2.5 py-1 text-[0.65rem] text-slate-300">
          Simulated client-side visualisation
        </span>
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between text-[0.7rem] text-slate-400">
          <span>Processing pipeline</span>
          <span>{Math.round(progress)}%</span>
        </div>
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
          <div
            className="h-full rounded-full bg-gradient-to-r from-emerald-400 via-accent to-sky-500 transition-[width]"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-2">
          <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
            Input frame
          </p>
          <div className="relative aspect-video overflow-hidden rounded-xl border border-slate-800 bg-slate-950/80">
            {inputUrl || activeSample ? (
              <Image
                src={inputUrl || activeSample?.inputPath || ""}
                alt="Input traffic frame"
                fill
                sizes="(max-width: 768px) 100vw, 50vw"
                className="object-cover"
              />
            ) : (
              <div className="flex h-full items-center justify-center text-[0.7rem] text-slate-500">
                Select a sample or upload a frame
              </div>
            )}
          </div>
        </div>

        <div className="space-y-2">
          <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
            VisionFlow visualisation
          </p>
          <div className="relative aspect-video overflow-hidden rounded-xl border border-slate-800 bg-slate-950/80">
            {isProcessing && (
              <div className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-2 bg-slate-950/60 backdrop-blur">
                <div className="h-10 w-10 animate-spin-slow rounded-full border-2 border-accent border-t-transparent" />
                <p className="text-[0.75rem] text-slate-200">
                  Running detection + plate reading…
                </p>
              </div>
            )}

            {showResult && activeSample && (
              <Image
                src={activeSample.resultPath}
                alt="VisionFlow processed output"
                fill
                sizes="(max-width: 768px) 100vw, 50vw"
                className="object-cover"
              />
            )}

            {showResult && !activeSample && inputUrl && (
              <>
                <Image
                  src={inputUrl}
                  alt="VisionFlow processed output"
                  fill
                  sizes="(max-width: 768px) 100vw, 50vw"
                  className="object-cover"
                />
                <div className="pointer-events-none absolute inset-0 z-10">
                  <div className="absolute left-3 top-3 rounded-xl bg-slate-900/80 px-3 py-2 text-[0.65rem] text-slate-100 shadow-glow">
                    <p className="font-semibold text-emerald-300">
                      VisionFlow detections
                    </p>
                    <p className="mt-0.5 text-[0.65rem] text-slate-300">
                      vehicles: 7 · plates: 3 · lanes: 3
                    </p>
                  </div>
                  <div className="absolute inset-6">
                    <div className="absolute left-[8%] top-[18%] h-[18%] w-[26%] rounded border border-emerald-400/90 shadow-[0_0_0_1px_rgba(16,185,129,0.5)]" />
                    <div className="absolute left-[42%] top-[30%] h-[20%] w-[28%] rounded border border-emerald-400/90 shadow-[0_0_0_1px_rgba(16,185,129,0.5)]" />
                    <div className="absolute left-[68%] top-[40%] h-[16%] w-[20%] rounded border border-emerald-400/90 shadow-[0_0_0_1px_rgba(16,185,129,0.5)]" />
                    <div className="absolute bottom-3 right-3 rounded-full bg-slate-900/85 px-3 py-1.5 text-[0.65rem] text-slate-200">
                      CSV output updated · lane load recomputed
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>

          {activeSample && (
            <p className="text-[0.7rem] text-slate-400">
              {activeSample.description}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

