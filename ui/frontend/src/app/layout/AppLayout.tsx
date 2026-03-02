import { NavLink } from "react-router-dom";
import type { PropsWithChildren } from "react";

const links = [
  { to: "/", label: "Dashboard" },
  { to: "/meetings", label: "Meetings" },
  { to: "/evaluation", label: "Evaluation" },
  { to: "/reproducibility", label: "Reproducibility" },
  { to: "/configuration", label: "Configuration" },
  { to: "/governance", label: "Governance" },
];

export function AppLayout({ children }: PropsWithChildren) {
  return (
    <div className="min-h-screen bg-transparent text-textPrimary">
      <header className="border-b border-accent/20 bg-accent text-white shadow-panel">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-6 lg:px-8">
          <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.28em] text-white/70">Northwestern-Style Research Console</p>
              <h1 className="text-3xl font-semibold tracking-tight md:text-4xl">Deterministic Meeting Intelligence</h1>
            </div>
            <div className="rounded-full border border-white/25 bg-white/10 px-4 py-2 text-sm text-white/90">
              Offline Mode ON
            </div>
          </div>
          <nav className="flex flex-wrap gap-2" aria-label="Primary">
            {links.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                end={link.to === "/"}
                className={({ isActive }) =>
                  [
                    "rounded-md border px-4 py-2 text-sm transition-colors",
                    isActive
                      ? "border-white/40 bg-white text-accent"
                      : "border-white/15 bg-white/10 text-white/80 hover:bg-white/18 hover:text-white",
                  ].join(" ")
                }
              >
                {link.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-6 lg:px-8">{children}</main>
    </div>
  );
}
