import Link from "next/link";

const navItems = [
  { href: "/", label: "Overview" },
  { href: "/demo", label: "Interactive Demo" },
  { href: "/results", label: "Results" },
  { href: "/architecture", label: "Architecture" }
];

export function Navbar() {
  return (
    <header className="sticky top-0 z-40 border-b border-slate-800/80 bg-slate-950/80 backdrop-blur">
      <nav className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3 sm:px-6 lg:px-8">
        <Link href="/" className="flex items-center gap-2 group">
          <div className="relative h-8 w-8 rounded-xl bg-gradient-to-tr from-accent via-indigo-500 to-sky-400 shadow-glow">
            <div className="absolute inset-1 rounded-lg bg-slate-950/70 grid-overlay" />
          </div>
          <div className="flex flex-col leading-tight">
            <span className="text-sm font-semibold tracking-widest text-slate-200">
              VISIONFLOW
            </span>
            <span className="text-xs text-slate-400 group-hover:text-slate-200 transition-colors">
              Traffic Vision System
            </span>
          </div>
        </Link>

        <div className="flex items-center gap-6">
          <ul className="hidden items-center gap-5 text-sm text-slate-300 md:flex">
            {navItems.map((item) => (
              <li key={item.href}>
                <Link
                  href={item.href}
                  className="hover:text-white transition-colors"
                >
                  {item.label}
                </Link>
              </li>
            ))}
          </ul>
          <a
            href="https://github.com/XreeceX/VisionFlow"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-900/70 px-3 py-1.5 text-xs font-medium text-slate-100 shadow-sm hover:border-accent hover:text-white hover:shadow-glow transition-all"
          >
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
            <span>View GitHub</span>
          </a>
        </div>
      </nav>
    </header>
  );
}

