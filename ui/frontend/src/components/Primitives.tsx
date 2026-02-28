import type { PropsWithChildren, ReactNode } from "react";
import type { StageState } from "../lib/types";

export function Panel({
  title,
  subtitle,
  actions,
  children,
  className = "",
}: PropsWithChildren<{
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  className?: string;
}>) {
  return (
    <section className={`rounded-xl border border-border bg-card/80 p-4 shadow-panel ${className}`}>
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-textPrimary">{title}</h2>
          {subtitle ? <p className="mt-1 text-sm text-textSecondary">{subtitle}</p> : null}
        </div>
        {actions}
      </div>
      {children}
    </section>
  );
}

export function StatusBadge({ state, label }: { state: StageState | "info"; label?: string }) {
  const tokens =
    state === "success"
      ? "border-success/40 bg-success/15 text-success"
      : state === "in_progress"
        ? "border-accent/40 bg-accent/15 text-accent"
      : state === "warn"
        ? "border-warning/40 bg-warning/15 text-warning"
        : state === "fail"
          ? "border-error/40 bg-error/15 text-error"
          : state === "not_run"
            ? "border-border bg-surface text-textSecondary"
            : "border-accent/40 bg-accent/15 text-accent";
  return (
    <span className={`inline-flex rounded-full border px-3 py-1 text-xs uppercase tracking-[0.2em] ${tokens}`}>
      {label ?? state}
    </span>
  );
}

export function KeyValueGrid({ items }: { items: Array<{ label: string; value: ReactNode }> }) {
  return (
    <dl className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
      {items.map((item) => (
        <div key={item.label} className="rounded-lg border border-border bg-surface/70 p-3">
          <dt className="text-xs uppercase tracking-[0.18em] text-textSecondary">{item.label}</dt>
          <dd className="mt-2 break-words text-sm text-textPrimary">{item.value}</dd>
        </div>
      ))}
    </dl>
  );
}

export function EmptyState({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-dashed border-border bg-surface/50 p-6 text-sm text-textSecondary">
      {message}
    </div>
  );
}
