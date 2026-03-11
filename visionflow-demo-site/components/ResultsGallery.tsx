import Image from "next/image";

type ResultItem = {
  id: string;
  inputPath: string;
  resultPath: string;
  title: string;
  description: string;
};

const RESULTS: ResultItem[] = [
  {
    id: "intersection-1",
    inputPath: "/sample-images/intersection-1.svg",
    resultPath: "/results/intersection-1-result.svg",
    title: "High-load intersection",
    description:
      "VisionFlow identifies multiple vehicles per lane, detects license plates, and exports aggregate counts to CSV."
  },
  {
    id: "emergency-vehicle",
    inputPath: "/sample-images/emergency-vehicle.svg",
    resultPath: "/results/emergency-vehicle-result.svg",
    title: "Emergency vehicle priority",
    description:
      "Emergency plate is detected and colour-coded, enabling downstream logic to prioritise the lane at the next signal cycle."
  },
  {
    id: "night-traffic",
    inputPath: "/sample-images/night-traffic.svg",
    resultPath: "/results/night-traffic-result.svg",
    title: "Low-light robustness",
    description:
      "Even in low-light conditions, the model can still localise vehicles and plates with confidence above the configured threshold."
  }
];

export function ResultsGallery() {
  return (
    <section className="space-y-5">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 className="text-xl font-semibold text-slate-50 sm:text-2xl">
            Example VisionFlow outputs
          </h2>
          <p className="mt-1 text-sm text-slate-400 max-w-xl">
            Each card pairs a raw input frame with the processed overlay that
            VisionFlow generates, along with a short description of what the
            system is capturing.
          </p>
        </div>
      </div>

      <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
        {RESULTS.map((item) => (
          <article
            key={item.id}
            className="glass-panel flex flex-col overflow-hidden rounded-2xl transition hover:-translate-y-0.5 hover:border-accent/60 hover:shadow-glow"
          >
            <div className="grid grid-cols-2 border-b border-slate-800/80 bg-slate-950/60">
              <div className="relative aspect-video border-r border-slate-800/80">
                <Image
                  src={item.inputPath}
                  alt={`${item.title} – input frame`}
                  fill
                  sizes="(max-width: 768px) 50vw, 33vw"
                  className="object-cover"
                />
              </div>
              <div className="relative aspect-video">
                <Image
                  src={item.resultPath}
                  alt={`${item.title} – VisionFlow output`}
                  fill
                  sizes="(max-width: 768px) 50vw, 33vw"
                  className="object-cover"
                />
              </div>
            </div>
            <div className="space-y-1.5 px-4 py-3">
              <h3 className="text-sm font-semibold text-slate-100">
                {item.title}
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                {item.description}
              </p>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

