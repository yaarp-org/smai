import { createFileRoute, Link } from "@tanstack/react-router";
import { z } from "zod";

import { LoadMore } from "@/components/common/load-more";
import { EmptyState, ErrorBanner, LoadingBlock } from "@/components/common/page-states";
import { CGCard } from "@/components/entity/cg-card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { $api } from "@/lib/api/client";

const cgStates = [
  "draft",
  "implementing",
  "implemented",
  "running",
  "evaluating",
  "complete",
  "implementation_failed",
  "running_failed",
  "evaluation_failed",
] as const;
type CGStateLiteral = (typeof cgStates)[number];

const cgListSearchSchema = z.object({
  state: z.enum(cgStates).optional(),
  proposal_id: z.string().optional(),
  cursor: z.string().optional(),
});

export const Route = createFileRoute("/comparison-groups")({
  validateSearch: cgListSearchSchema,
  component: CGListPage,
});

function CGListPage() {
  const search = Route.useSearch();
  const navigate = Route.useNavigate();

  const query = $api.useQuery("get", "/api/v1/comparison-groups", {
    params: {
      query: {
        state: search.state ?? null,
        proposal_id: search.proposal_id ?? null,
        cursor: search.cursor ?? null,
      },
    },
  });

  return (
    <div className="space-y-4">
      <header className="space-y-1">
        <h1 className="text-2xl font-bold tracking-tight">Comparison Groups</h1>
        <p className="text-sm text-[var(--color-fg-subtle)]">
          The CG-execution pipeline tracks each comparison group from draft → complete (or failed).
        </p>
      </header>

      <div className="flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-2 text-sm">
          <span className="text-[var(--color-fg-subtle)]">State</span>
          <Select
            value={search.state ?? "__all__"}
            onValueChange={(value) => {
              const next: CGStateLiteral | undefined =
                value === "__all__" ? undefined : (value as CGStateLiteral);
              void navigate({
                search: { state: next, proposal_id: search.proposal_id, cursor: undefined },
              });
            }}
          >
            <SelectTrigger className="h-9 w-52">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__all__">All states</SelectItem>
              {cgStates.map((s) => (
                <SelectItem key={s} value={s}>
                  {s}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </label>
        <label className="flex items-center gap-2 text-sm">
          <span className="text-[var(--color-fg-subtle)]">Proposal ID</span>
          <Input
            value={search.proposal_id ?? ""}
            onChange={(e) =>
              void navigate({
                search: {
                  state: search.state,
                  proposal_id: e.target.value || undefined,
                  cursor: undefined,
                },
              })
            }
            placeholder="prop_…"
            className="h-9 w-64 font-mono"
          />
        </label>
        {(search.state || search.proposal_id || search.cursor) && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() =>
              void navigate({
                search: { state: undefined, proposal_id: undefined, cursor: undefined },
              })
            }
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
            <EmptyState message="No comparison groups match the current filters." />
          ) : (
            <ul className="space-y-2">
              {query.data.items.map((cg) => (
                <li key={cg.id}>
                  <CGCard cg={cg} />
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
                to="/comparison-groups"
                search={{
                  state: search.state,
                  proposal_id: search.proposal_id,
                  cursor: undefined,
                }}
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
