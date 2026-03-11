export default function ArchitecturePage() {
  return (
    <div className="py-8 space-y-7">
      <div className="max-w-xl">
        <h1 className="text-2xl font-semibold text-slate-50 sm:text-3xl">
          Architecture
        </h1>
        <p className="mt-2 text-sm text-slate-400">
          VisionFlow is a Python-based traffic analytics pipeline that ingests
          video, runs computer-vision models, and exports both visual overlays
          and structured CSV logs. This page explains the major stages.
        </p>
      </div>

      <section className="glass-panel rounded-2xl p-5 sm:p-6 space-y-4">
        <h2 className="text-lg font-semibold text-slate-50">Data flow</h2>
        <p className="text-sm text-slate-300">
          Video frames are read from a traffic camera or sample video file. For
          each frame, VisionFlow performs vehicle and license plate detection,
          OCR, and metadata extraction before exporting both an annotated frame
          and a row in{" "}
          <span className="font-mono text-xs text-slate-200">
            TrafficRecords.csv
          </span>
          .
        </p>
        <div className="mt-3 grid gap-3 text-xs text-slate-300 sm:grid-cols-3">
          <div className="rounded-xl bg-slate-950/80 p-3 border border-slate-800">
            <p className="font-semibold text-slate-100 mb-1">Input</p>
            <p>Traffic video stream, captured frame-by-frame via OpenCV.</p>
          </div>
          <div className="rounded-xl bg-slate-950/80 p-3 border border-slate-800">
            <p className="font-semibold text-slate-100 mb-1">Processing</p>
            <p>YOLOv5 detects vehicles and plates, EasyOCR reads plate text.</p>
          </div>
          <div className="rounded-xl bg-slate-950/80 p-3 border border-slate-800">
            <p className="font-semibold text-slate-100 mb-1">Outputs</p>
            <p>Annotated frames and CSV rows for every detection event.</p>
          </div>
        </div>
      </section>

      <section className="glass-panel rounded-2xl p-5 sm:p-6 space-y-4">
        <h2 className="text-lg font-semibold text-slate-50">
          Model and algorithm
        </h2>
        <ul className="space-y-2 text-sm text-slate-300">
          <li>
            <span className="font-semibold text-slate-100">YOLOv5</span> is
            used as the object detector to localise vehicles and license plates
            within each frame.
          </li>
          <li>
            <span className="font-semibold text-slate-100">EasyOCR</span> runs
            on the cropped plate region to recover the plate text.
          </li>
          <li>
            <span className="font-semibold text-slate-100">Plate colour</span>{" "}
            is derived from pixel statistics on the plate patch and used to
            infer vehicle class (eg. commercial vs. private vs. emergency).
          </li>
          <li>
            <span className="font-semibold text-slate-100">
              Lane assignment
            </span>{" "}
            is simulated, giving each detection a lane identifier so that
            VisionFlow can reason in terms of per-lane volume.
          </li>
        </ul>
      </section>

      <section className="glass-panel rounded-2xl p-5 sm:p-6 space-y-4">
        <h2 className="text-lg font-semibold text-slate-50">
          Processing pipeline
        </h2>
        <ol className="grid gap-3 text-sm text-slate-300 sm:grid-cols-5">
          <li className="rounded-xl bg-slate-950/80 p-3 border border-slate-800">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              1. Input acquisition
            </p>
            <p className="mt-1 text-xs">
              Frames are pulled from a video source via OpenCV.
            </p>
          </li>
          <li className="rounded-xl bg-slate-950/80 p-3 border border-slate-800">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              2. Preprocessing
            </p>
            <p className="mt-1 text-xs">
              Resolution, colour space, and batching are normalised for YOLOv5.
            </p>
          </li>
          <li className="rounded-xl bg-slate-950/80 p-3 border border-slate-800">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              3. VisionFlow model
            </p>
            <p className="mt-1 text-xs">
              YOLOv5 locates vehicles and plates; EasyOCR reads plate text.
            </p>
          </li>
          <li className="rounded-xl bg-slate-950/80 p-3 border border-slate-800">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              4. Post-processing
            </p>
            <p className="mt-1 text-xs">
              Plate colour, lane assignment, confidence thresholds, and
              filtering are applied.
            </p>
          </li>
          <li className="rounded-xl bg-slate-950/80 p-3 border border-slate-800">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              5. Visualisation
            </p>
            <p className="mt-1 text-xs">
              Annotated frames and CSV rows are written out for dashboards,
              monitoring, or control logic.
            </p>
          </li>
        </ol>
      </section>
    </div>
  );
}

