import { Link } from "@tanstack/react-router";

import { StateBadge } from "@/components/entity/state-badge";
import { formatRelative } from "@/lib/format/datetime";
import type { components } from "@/lib/api/generated/api-types";

type PaperSummary = components["schemas"]["PaperSummary"];

export function PaperCard({ paper }: { paper: PaperSummary }) {
  return (
    <Link
      to="/papers/$arxivId"
      params={{ arxivId: paper.arxiv_id }}
      className="block rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] p-4 transition-colors hover:bg-[var(--color-bg-subtle)]"
    >
      <div className="flex flex-wrap items-center gap-2">
        <code className="font-mono text-sm font-medium">{paper.arxiv_id}</code>
        <StateBadge state={paper.state} />
        <span className="ml-auto text-xs text-[var(--color-fg-subtle)]" title={paper.updated_at}>
          updated {formatRelative(paper.updated_at)}
        </span>
      </div>
      {paper.title && <div className="mt-2 line-clamp-2 text-sm">{paper.title}</div>}
      {paper.last_error && (
        <div className="mt-2 line-clamp-2 text-xs text-[var(--color-danger)]">
          last error: {paper.last_error}
        </div>
      )}
      {paper.state === "partial" && (
        <div className="mt-2 text-xs text-[var(--color-warning)]">
          partial — promote to register the paper.
        </div>
      )}
    </Link>
  );
}
