import { Hero } from "@/components/Hero";
import { PipelineDiagram } from "@/components/PipelineDiagram";
import Link from "next/link";

const techStack = [
  { label: "Python", detail: "Core CV + ML implementation" },
  { label: "Computer Vision", detail: "YOLOv5, OpenCV, plate tracking" },
  { label: "Machine Learning", detail: "Detection & recognition models" },
  { label: "Next.js", detail: "Interactive portfolio frontend" },
  { label: "Vercel", detail: "Serverless deployment for this demo" }
];

export default function HomePage() {
  return (
    <div className="py-4 sm:py-6">
      <Hero />

      <section className="mb-12 grid gap-6 md:grid-cols-[1.4fr,1fr]">
        <div className="glass-panel rounded-2xl p-5 sm:p-6">
          <h2 className="text-lg font-semibold text-slate-50 sm:text-xl">
            Problem
          </h2>
          <p className="mt-3 text-sm text-slate-300 leading-relaxed">
            Urban intersections generate a constant stream of vehicles, but most
            traffic lights still operate on fixed timers. Emergency vehicles
            wait in queues, high-load lanes get under-served, and valuable
            license plate and traffic pattern data is lost.
          </p>
          <p className="mt-3 text-sm text-slate-300 leading-relaxed">
            VisionFlow turns raw traffic video into structured insights:
            detecting vehicles, reading license plates, classifying plate color,
            and assigning lanes in real-time. Those signals can drive smarter,
            adaptive traffic control and downstream analytics.
          </p>
        </div>

        <div className="glass-panel rounded-2xl p-5 sm:p-6 space-y-3">
          <h3 className="text-sm font-medium text-slate-100">
            What this demo shows
          </h3>
          <p className="text-xs text-slate-300">
            This site visualizes the{" "}
            <span className="font-semibold text-slate-100">
              VisionFlow processing pipeline and sample outputs
            </span>{" "}
            from the Python backend project.
          </p>
          <ul className="space-y-1.5 text-xs text-slate-300">
            <li>• Step-by-step pipeline explanation</li>
            <li>• Interactive mock demo with example frames</li>
            <li>• Gallery of input/output pairs</li>
            <li>• Architecture and data flow overview</li>
          </ul>
          <Link
            href="/architecture"
            className="inline-flex text-xs font-medium text-accent hover:text-indigo-300 transition-colors"
          >
            Explore architecture →
          </Link>
        </div>
      </section>

      <PipelineDiagram />

      <section className="mb-16">
        <div className="flex items-end justify-between gap-2">
          <div>
            <h2 className="text-lg font-semibold text-slate-50 sm:text-xl">
              Tech stack
            </h2>
            <p className="mt-1 text-sm text-slate-400 max-w-xl">
              Production pipeline in Python and CV, wrapped in a modern
              Next.js/Tailwind demo for showcasing the work.
            </p>
          </div>
        </div>

        <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {techStack.map((item) => (
            <div
              key={item.label}
              className="glass-panel rounded-xl p-4 transition hover:-translate-y-0.5 hover:border-accent/70 hover:shadow-glow"
            >
              <p className="text-sm font-semibold text-slate-100">
                {item.label}
              </p>
              <p className="mt-1 text-xs text-slate-400">{item.detail}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

