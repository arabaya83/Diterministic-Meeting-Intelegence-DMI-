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
      <header className="border-b border-border/80 bg-surface/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-6 lg:px-8">
          <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
            <div>
              <h1 className="text-3xl font-semibold tracking-tight">Deterministic Meeting Intelligence (DMI)</h1>
            </div>
            <div className="rounded-full border border-success/50 bg-success/10 px-4 py-2 text-sm text-success">
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
                      ? "border-accent bg-accent/20 text-textPrimary"
                      : "border-border bg-card/70 text-textSecondary hover:bg-card hover:text-textPrimary",
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
