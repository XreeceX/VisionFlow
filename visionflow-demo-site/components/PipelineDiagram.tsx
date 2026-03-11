type Stage = {
  title: string;
  subtitle: string;
  badge: string;
};

const stages: Stage[] = [
  {
    title: "Input",
    subtitle: "Traffic video or camera frames",
    badge: "1"
  },
  {
    title: "Preprocessing",
    subtitle: "Resize, normalize, prepare for inference",
    badge: "2"
  },
  {
    title: "VisionFlow Model",
    subtitle: "YOLOv5 + EasyOCR + plate color logic",
    badge: "3"
  },
  {
    title: "Output",
    subtitle: "Annotated frames + CSV traffic records",
    badge: "4"
  }
];

export function PipelineDiagram() {
  return (
    <section className="mb-12 space-y-4">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 className="text-xl font-semibold text-slate-50 sm:text-2xl">
            VisionFlow pipeline
          </h2>
          <p className="text-sm text-slate-400 max-w-xl mt-1">
            From raw traffic footage to structured, queryable traffic records
            and visual overlays.
          </p>
        </div>
      </div>

      <div className="mt-4 overflow-x-auto pb-3">
        <div className="flex min-w-[640px] items-stretch justify-between gap-4 rounded-2xl border border-slate-800 bg-slate-950/70 px-4 py-4 sm:px-6 sm:py-5">
          {stages.map((stage, index) => (
            <div
              key={stage.title}
              className="flex flex-1 flex-col items-center text-center"
            >
              <div className="relative mb-3 flex items-center justify-center">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-slate-900 border border-slate-700 shadow-sm">
                  <span className="text-xs font-semibold text-slate-200">
                    {stage.badge}
                  </span>
                </div>
                {index < stages.length - 1 && (
                  <div className="absolute left-1/2 right-[-50%] top-1/2 -z-10 h-px -translate-y-1/2 bg-gradient-to-r from-slate-700 via-accent/60 to-slate-700" />
                )}
              </div>
              <h3 className="text-sm font-medium text-slate-100">
                {stage.title}
              </h3>
              <p className="mt-1 text-xs text-slate-400">{stage.subtitle}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

