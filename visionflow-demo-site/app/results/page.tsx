import { ResultsGallery } from "@/components/ResultsGallery";

export default function ResultsPage() {
  return (
    <div className="py-8 space-y-6">
      <div className="max-w-xl">
        <h1 className="text-2xl font-semibold text-slate-50 sm:text-3xl">
          Results gallery
        </h1>
        <p className="mt-2 text-sm text-slate-400">
          A curated set of input / output pairs that illustrate how VisionFlow
          sees traffic scenes, detects license plates, and provides structured
          data for downstream logic.
        </p>
      </div>

      <ResultsGallery />
    </div>
  );
}

