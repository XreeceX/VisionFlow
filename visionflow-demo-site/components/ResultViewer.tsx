"use client";

import Image from "next/image";
import { useEffect, useMemo, useState } from "react";
import type { DemoSample } from "./DemoUploader";

type ResultViewerProps = {
  activeSample: DemoSample | null;
  inputUrl: string | null;
  isProcessing: boolean;
};

type GeneratedOverlay = {
  timestamp: string;
  plate: string;
  color: "Red" | "Yellow" | "White";
  lane: "R1" | "R2" | "R3" | "R4";
  vehicleType: "car" | "truck" | "bus" | "motorbike";
  confidence: number;
};

function generateOverlayFromUrl(url: string): GeneratedOverlay {
  let hash = 0;
  for (let i = 0; i < url.length; i += 1) {
    hash = (hash * 31 + url.charCodeAt(i)) >>> 0;
  }

  const colors: GeneratedOverlay["color"][] = ["Red", "Yellow", "White"];
  const lanes: GeneratedOverlay["lane"][] = ["R1", "R2", "R3", "R4"];
  const vehicles: GeneratedOverlay["vehicleType"][] = [
    "car",
    "truck",
    "bus",
    "motorbike"
  ];

  const plateNumber = 1000 + (hash % 8999);
  const plate = `PLT-${plateNumber}`;
  const color = colors[hash % colors.length];
  const lane = lanes[(hash >> 3) % lanes.length];
  const vehicleType = vehicles[(hash >> 5) % vehicles.length];
  const confidence = Math.round((0.8 + (hash % 20) / 100) * 100) / 100;

  const date = new Date();
  const timestamp = date.toISOString().replace("T", " ").slice(0, 19);

  return { timestamp, plate, color, lane, vehicleType, confidence };
}

export function ResultViewer({
  activeSample,
  inputUrl,
  isProcessing
}: ResultViewerProps) {
  const [progress, setProgress] = useState(0);
  const overlay = useMemo(
    () => (inputUrl && !activeSample ? generateOverlayFromUrl(inputUrl) : null),
    [inputUrl, activeSample]
  );

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
          <div className="relative aspect-video rounded-xl border border-slate-800 bg-slate-950/80 px-4 py-3">
            {activeSample ? (
              <div className="relative h-full w-full rounded-lg bg-slate-950/80">
                <div className="relative mx-auto h-[88%] w-[92%] overflow-hidden rounded-lg border border-slate-800/80">
                  <Image
                    src={activeSample.inputPath}
                    alt="Input traffic frame"
                    fill
                    sizes="(max-width: 768px) 100vw, 50vw"
                    className="object-contain"
                  />
                </div>
              </div>
            ) : (
              <div className="flex h-full items-center justify-center text-[0.7rem] text-slate-500">
                Choose a sample to preview
              </div>
            )}
          </div>
        </div>

        <div className="space-y-2">
          <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
            VisionFlow visualisation
          </p>
          <div className="relative aspect-video rounded-xl border border-slate-800 bg-slate-950/80 px-4 py-3">
            {isProcessing && (
              <div className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-2 bg-slate-950/60 backdrop-blur">
                <div className="h-10 w-10 animate-spin-slow rounded-full border-2 border-accent border-t-transparent" />
                <p className="text-[0.75rem] text-slate-200">
                  Running detection + plate reading…
                </p>
              </div>
            )}

            {showResult && activeSample && (
              <div className="relative h-full w-full rounded-lg bg-slate-950/80">
                <div className="relative mx-auto h-[88%] w-[92%] overflow-hidden rounded-lg border border-slate-800/80">
                  <Image
                    src={activeSample.resultPath}
                    alt="VisionFlow processed output"
                    fill
                    sizes="(max-width: 768px) 100vw, 50vw"
                    className="object-contain"
                  />
                </div>
              </div>
            )}

            {showResult && !activeSample && inputUrl && overlay && (
              <>
                <Image
                  src={inputUrl}
                  alt="VisionFlow processed output"
                  fill
                  sizes="(max-width: 768px) 100vw, 50vw"
                  className="object-cover"
                />
                <div className="pointer-events-none absolute inset-0 z-10">
                  <div className="absolute inset-5">
                    {/* lane signals – matches VisionFlow Python overlay */}
                    <div className="absolute left-2 top-2 space-y-1 text-[0.7rem] font-semibold text-slate-100 drop-shadow">
                      <p>
                        R1: <span className="text-emerald-400">🟢</span>
                      </p>
                      <p>
                        R2: <span className="text-rose-400">🔴</span>
                      </p>
                      <p>
                        R3: <span className="text-rose-400">🔴</span>
                      </p>
                      <p>
                        R4: <span className="text-rose-400">🔴</span>
                      </p>
                    </div>

                    {/* detection boxes with plate + colour labels */}
                    <div className="absolute left-[10%] top-[20%] h-[18%] w-[24%] rounded border-2 border-emerald-400 shadow-[0_0_0_1px_rgba(16,185,129,0.7)]">
                      <div className="absolute -top-5 left-0 rounded-md bg-emerald-500/95 px-1.5 py-0.5 text-[0.55rem] font-semibold text-slate-900">
                        {overlay.plate} ({overlay.color})
                      </div>
                      <div className="absolute -bottom-5 left-0 text-[0.55rem] font-semibold text-emerald-300">
                        {overlay.vehicleType} · conf {overlay.confidence.toFixed(2)}
                      </div>
                    </div>
                    <div className="absolute left-[45%] top-[33%] h-[20%] w-[26%] rounded border-2 border-emerald-400/70 shadow-[0_0_0_1px_rgba(16,185,129,0.5)]" />
                    <div className="absolute left-[70%] top-[42%] h-[16%] w-[18%] rounded border-2 border-emerald-400/50 shadow-[0_0_0_1px_rgba(16,185,129,0.4)]" />

                    {/* CSV-style summary using real CSV field names */}
                    <div className="absolute bottom-3 left-3 rounded-xl bg-slate-950/85 px-3 py-2 text-[0.6rem] text-slate-100 backdrop-blur shadow-glow">
                      <div className="grid grid-cols-6 gap-3">
                        <div>
                          <p className="text-[0.55rem] uppercase tracking-wide text-slate-400">
                            Timestamp
                          </p>
                          <p>{overlay.timestamp}</p>
                        </div>
                        <div>
                          <p className="text-[0.55rem] uppercase tracking-wide text-slate-400">
                            LicensePlate
                          </p>
                          <p>{overlay.plate}</p>
                        </div>
                        <div>
                          <p className="text-[0.55rem] uppercase tracking-wide text-slate-400">
                            PlateColor
                          </p>
                          <p>{overlay.color}</p>
                        </div>
                        <div>
                          <p className="text-[0.55rem] uppercase tracking-wide text-slate-400">
                            Lane
                          </p>
                          <p>{overlay.lane}</p>
                        </div>
                        <div>
                          <p className="text-[0.55rem] uppercase tracking-wide text-slate-400">
                            VehicleType
                          </p>
                          <p>{overlay.vehicleType}</p>
                        </div>
                        <div>
                          <p className="text-[0.55rem] uppercase tracking-wide text-slate-400">
                            Confidence
                          </p>
                          <p>{overlay.confidence.toFixed(2)}</p>
                        </div>
                      </div>
                      <p className="mt-1 text-[0.55rem] text-slate-400">
                        Fields mirror rows written to TrafficRecords.csv in the Python VisionFlow pipeline.
                      </p>
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

