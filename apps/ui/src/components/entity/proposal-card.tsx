import { Link } from "@tanstack/react-router";

import { StateBadge } from "@/components/entity/state-badge";
import { formatRelative } from "@/lib/format/datetime";
import type { components } from "@/lib/api/generated/api-types";

type ProposalSummary = components["schemas"]["ProposalSummary"];

export function ProposalCard({ proposal }: { proposal: ProposalSummary }) {
  return (
    <Link
      to="/proposals/$id"
      params={{ id: proposal.id }}
      className="block rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] p-4 transition-colors hover:bg-[var(--color-bg-subtle)]"
    >
      <div className="flex flex-wrap items-center gap-2">
        <code className="font-mono text-sm font-medium">{proposal.id}</code>
        <StateBadge state={proposal.state} />
        <span className="rounded bg-[var(--color-bg-subtle)] px-1.5 py-0.5 font-mono text-xs text-[var(--color-fg-subtle)]">
          {proposal.submission_kind}
        </span>
        <span className="ml-auto text-xs text-[var(--color-fg-subtle)]" title={proposal.updated_at}>
          updated {formatRelative(proposal.updated_at)}
        </span>
      </div>
      {proposal.reproduce_paper_arxiv_id && (
        <div className="mt-2 text-xs text-[var(--color-fg-subtle)]">
          reproducing arXiv:
          <code className="ml-1 font-mono">{proposal.reproduce_paper_arxiv_id}</code>
        </div>
      )}
      {proposal.last_error && (
        <div className="mt-2 line-clamp-2 text-xs text-[var(--color-danger)]">
          last error: {proposal.last_error}
        </div>
      )}
    </Link>
  );
}
