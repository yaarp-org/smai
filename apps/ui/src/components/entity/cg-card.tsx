import { Link } from "@tanstack/react-router";

import { StateBadge } from "@/components/entity/state-badge";
import { formatRelative } from "@/lib/format/datetime";
import type { components } from "@/lib/api/generated/api-types";

type CGSummary = components["schemas"]["CGSummary"];

export function CGCard({ cg }: { cg: CGSummary }) {
  return (
    <Link
      to="/comparison-groups/$id"
      params={{ id: cg.id }}
      className="block rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] p-4 transition-colors hover:bg-[var(--color-bg-subtle)]"
    >
      <div className="flex flex-wrap items-center gap-2">
        <code className="font-mono text-sm font-medium">{cg.id}</code>
        <StateBadge state={cg.state} />
        {cg.is_terminal && (
          <span className="rounded bg-[var(--color-bg-subtle)] px-1.5 py-0.5 text-xs text-[var(--color-fg-subtle)]">
            terminal
          </span>
        )}
        <span className="ml-auto text-xs text-[var(--color-fg-subtle)]" title={cg.updated_at}>
          updated {formatRelative(cg.updated_at)}
        </span>
      </div>
      <div className="mt-2 text-xs text-[var(--color-fg-subtle)]">
        proposal: <code className="font-mono">{cg.proposal_id}</code>
      </div>
      {cg.last_error && (
        <div className="mt-2 line-clamp-2 text-xs text-[var(--color-danger)]">
          last error: {cg.last_error}
        </div>
      )}
    </Link>
  );
}
