import { cn } from "@/lib/utils";

// 13-frontend.md §9.1 design tokens. Each state literal maps to one of three
// visual treatments:
//   pending  → fg-subtle (neutral; not yet picked up)
//   in-flight → accent (worker is actively driving the entity)
//   success  → success (terminal accept)
//   failure  → danger (terminal reject / failed)
//
// All entity state literals from `_common.py` (proposal / paper / CG / entry /
// run) are enumerated here. New states added to the API contract surface as a
// TS error since the union narrows in `tone()`.

type StateTone = "pending" | "active" | "success" | "danger" | "warning";

const STATE_TONE: Record<string, StateTone> = {
  // Proposal
  proposal_submitted: "pending",
  designing: "active",
  designed: "warning",
  registered: "success",
  rejected: "danger",
  failed: "danger",
  // Paper
  submitted: "pending",
  fetching: "active",
  screening: "active",
  planning: "active",
  partial: "warning",
  // CG
  draft: "pending",
  implementing: "active",
  implemented: "active",
  running: "active",
  evaluating: "active",
  complete: "success",
  implementation_failed: "danger",
  running_failed: "danger",
  evaluation_failed: "danger",
  // Entry
  pending: "pending",
  // Run
  succeeded: "success",
  inconclusive: "warning",
};

const TONE_CLASSES: Record<StateTone, string> = {
  pending:
    "bg-[color-mix(in_oklch,var(--color-state-pending)_15%,transparent)] text-[var(--color-state-pending)] ring-[color-mix(in_oklch,var(--color-state-pending)_30%,transparent)]",
  active:
    "bg-[color-mix(in_oklch,var(--color-state-running)_15%,transparent)] text-[var(--color-state-running)] ring-[color-mix(in_oklch,var(--color-state-running)_30%,transparent)]",
  success:
    "bg-[color-mix(in_oklch,var(--color-state-complete)_15%,transparent)] text-[var(--color-state-complete)] ring-[color-mix(in_oklch,var(--color-state-complete)_30%,transparent)]",
  danger:
    "bg-[color-mix(in_oklch,var(--color-state-failed)_15%,transparent)] text-[var(--color-state-failed)] ring-[color-mix(in_oklch,var(--color-state-failed)_30%,transparent)]",
  warning:
    "bg-[color-mix(in_oklch,var(--color-warning)_15%,transparent)] text-[var(--color-warning)] ring-[color-mix(in_oklch,var(--color-warning)_30%,transparent)]",
};

export interface StateBadgeProps {
  state: string;
  className?: string;
}

export function StateBadge({ state, className }: StateBadgeProps) {
  const tone: StateTone = STATE_TONE[state] ?? "pending";
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2 py-0.5 font-mono text-xs font-medium ring-1 ring-inset",
        TONE_CLASSES[tone],
        className,
      )}
    >
      {state}
    </span>
  );
}
