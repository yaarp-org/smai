import { createFileRoute, Link } from "@tanstack/react-router";
import { z } from "zod";

import { LoadMore } from "@/components/common/load-more";
import { EmptyState, ErrorBanner, LoadingBlock } from "@/components/common/page-states";
import { ProposalCard } from "@/components/entity/proposal-card";
import { Button, buttonVariants } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { $api } from "@/lib/api/client";

// 13-frontend.md §6.2 / §8.2: typed search-params via Zod. The state values
// here mirror the ProposalState literal in smai_api_spec/_common.py. URL
// search-param state survives refresh and Back/Forward navigation.
const proposalStates = [
  "proposal_submitted",
  "designing",
  "designed",
  "registered",
  "rejected",
  "failed",
] as const;
type ProposalStateLiteral = (typeof proposalStates)[number];

const proposalListSearchSchema = z.object({
  state: z.enum(proposalStates).optional(),
  cursor: z.string().optional(),
});

export const Route = createFileRoute("/proposals")({
  validateSearch: proposalListSearchSchema,
  component: ProposalsListPage,
});

function ProposalsListPage() {
  const search = Route.useSearch();
  const navigate = Route.useNavigate();

  const query = $api.useQuery("get", "/api/v1/proposals", {
    params: {
      query: {
        state: search.state ?? null,
        cursor: search.cursor ?? null,
      },
    },
  });

  return (
    <div className="space-y-4">
      <header className="flex flex-wrap items-center gap-3">
        <div className="flex-1 space-y-1">
          <h1 className="text-2xl font-bold tracking-tight">Proposals</h1>
          <p className="text-sm text-[var(--color-fg-subtle)]">
            Primary input verb per DEC-032 — every user-driven CG creation flows through here.
          </p>
        </div>
        <Link to="/proposals/new" className={buttonVariants()}>
          New proposal
        </Link>
      </header>

      <div className="flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-2 text-sm">
          <span className="text-[var(--color-fg-subtle)]">State</span>
          <Select
            value={search.state ?? "__all__"}
            onValueChange={(value) => {
              const next: ProposalStateLiteral | undefined =
                value === "__all__" ? undefined : (value as ProposalStateLiteral);
              void navigate({ search: { state: next, cursor: undefined } });
            }}
          >
            <SelectTrigger className="h-9 w-44">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__all__">All states</SelectItem>
              {proposalStates.map((s) => (
                <SelectItem key={s} value={s}>
                  {s}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </label>
        {(search.state || search.cursor) && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => void navigate({ search: { state: undefined, cursor: undefined } })}
          >
            Clear filters
          </Button>
        )}
      </div>

      {query.isPending && <LoadingBlock />}
      {query.error && <ErrorBanner error={query.error} />}

      {query.data && (
        <>
          {query.data.items.length === 0 ? (
            <EmptyState message="No proposals match the current filters." />
          ) : (
            <ul className="space-y-2">
              {query.data.items.map((proposal) => (
                <li key={proposal.id}>
                  <ProposalCard proposal={proposal} />
                </li>
              ))}
            </ul>
          )}
          <LoadMore
            nextCursor={query.data.next_cursor}
            onLoadMore={(cursor) => void navigate({ search: (prev) => ({ ...prev, cursor }) })}
            loading={query.isFetching}
          />
          {search.cursor && (
            <div className="text-center">
              <Link
                to="/proposals"
                search={{ state: search.state, cursor: undefined }}
                className="text-xs text-[var(--color-fg-subtle)] hover:underline"
              >
                Back to first page
              </Link>
            </div>
          )}
        </>
      )}
    </div>
  );
}
