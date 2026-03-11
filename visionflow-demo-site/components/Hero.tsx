import Link from "next/link";

export function Hero() {
  return (
    <section className="relative pt-10 pb-14 sm:pt-14 sm:pb-20">
      <div className="pointer-events-none absolute inset-0 -z-10 opacity-60">
        <div className="grid-overlay absolute inset-0" />
        <div className="absolute -top-32 left-1/2 h-64 w-64 -translate-x-1/2 rounded-full bg-accent/40 blur-3xl" />
      </div>

      <div className="flex flex-col items-start gap-8 md:flex-row md:items-center md:justify-between">
        <div className="max-w-xl space-y-5">
          <span className="inline-flex items-center gap-2 rounded-full border border-emerald-500/40 bg-emerald-500/10 px-3 py-1 text-[0.7rem] font-medium text-emerald-300">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
            Real-time traffic vision pipeline
          </span>

          <div className="space-y-3">
            <h1 className="text-3xl font-semibold tracking-tight text-slate-50 sm:text-4xl lg:text-5xl">
              VisionFlow
            </h1>
            <p className="text-base text-slate-300 sm:text-lg">
              An intelligent computer-vision system that understands traffic
              flows, detects vehicles and license plates, and powers smarter
              signal control.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <Link
              href="/demo"
              className="inline-flex items-center justify-center rounded-full bg-gradient-to-r from-accent via-indigo-500 to-sky-500 px-5 py-2.5 text-sm font-medium text-white shadow-glow transition hover:brightness-110"
            >
              Try interactive demo
            </Link>
            <a
              href="https://github.com/XreeceX/VisionFlow"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center justify-center gap-2 rounded-full border border-slate-700 bg-slate-900/70 px-4 py-2.5 text-sm font-medium text-slate-100 hover:border-accent hover:shadow-glow transition"
            >
              <span className="i-tabler-brand-github h-4 w-4" aria-hidden />
              <span>View code on GitHub</span>
            </a>
          </div>

          <p className="text-xs text-slate-400 max-w-md">
            This site is a portfolio-friendly, fully client-side demo of the{" "}
            <span className="font-semibold text-slate-200">
              VisionFlow LPR &amp; traffic analytics pipeline
            </span>
            , built for recruiters and collaborators.
          </p>
        </div>

        <div className="w-full max-w-md md:max-w-sm lg:max-w-md">
          <div className="glass-panel relative overflow-hidden rounded-2xl px-4 py-4 sm:px-6 sm:py-6">
            <div className="mb-3 flex items-center justify-between text-xs text-slate-400">
              <span className="flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                Live pipeline overview
              </span>
              <span>~30 FPS</span>
            </div>
            <div className="space-y-3 text-xs sm:text-sm">
              <div className="flex items-center justify-between rounded-xl bg-slate-900/80 p-3">
                <span className="text-slate-300">Traffic video frames</span>
                <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[0.65rem] text-slate-400">
                  Input
                </span>
              </div>
              <div className="flex items-center justify-between rounded-xl bg-slate-900/80 p-3">
                <span className="text-slate-300">
                  YOLOv5 vehicle + plate detection
                </span>
                <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[0.65rem] text-slate-400">
                  Detection
                </span>
              </div>
              <div className="flex items-center justify-between rounded-xl bg-slate-900/80 p-3">
                <span className="text-slate-300">
                  EasyOCR plate reading &amp; color logic
                </span>
                <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[0.65rem] text-slate-400">
                  Semantics
                </span>
              </div>
              <div className="flex items-center justify-between rounded-xl bg-slate-900/80 p-3">
                <span className="text-slate-300">
                  Lane assignment + signal prioritization
                </span>
                <span className="rounded-full bg-emerald-500/20 px-2 py-0.5 text-[0.65rem] text-emerald-300">
                  Decisions
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

