import { createFileRoute, Link } from "@tanstack/react-router";
import { z } from "zod";

import { LoadMore } from "@/components/common/load-more";
import { EmptyState, ErrorBanner, LoadingBlock } from "@/components/common/page-states";
import { RunRow } from "@/components/entity/run-row";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Table, TableBody, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { $api } from "@/lib/api/client";

const runStates = [
  "pending",
  "submitted",
  "running",
  "succeeded",
  "failed",
  "inconclusive",
] as const;
type RunStateLiteral = (typeof runStates)[number];

const runListSearchSchema = z.object({
  state: z.enum(runStates).optional(),
  cg_id: z.string().optional(),
  entry_id: z.string().optional(),
  cursor: z.string().optional(),
});

export const Route = createFileRoute("/runs")({
  validateSearch: runListSearchSchema,
  component: RunsListPage,
});

function RunsListPage() {
  const search = Route.useSearch();
  const navigate = Route.useNavigate();

  const query = $api.useQuery("get", "/api/v1/runs", {
    params: {
      query: {
        state: search.state ?? null,
        cg_id: search.cg_id ?? null,
        entry_id: search.entry_id ?? null,
        cursor: search.cursor ?? null,
      },
    },
  });

  return (
    <div className="space-y-4">
      <header className="space-y-1">
        <h1 className="text-2xl font-bold tracking-tight">Runs</h1>
        <p className="text-sm text-[var(--color-fg-subtle)]">
          Cross-CG run list. Each run is one (entry × seed) pair dispatched by the worker.
        </p>
      </header>

      <div className="flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-2 text-sm">
          <span className="text-[var(--color-fg-subtle)]">State</span>
          <Select
            value={search.state ?? "__all__"}
            onValueChange={(value) => {
              const next: RunStateLiteral | undefined =
                value === "__all__" ? undefined : (value as RunStateLiteral);
              void navigate({
                search: {
                  state: next,
                  cg_id: search.cg_id,
                  entry_id: search.entry_id,
                  cursor: undefined,
                },
              });
            }}
          >
            <SelectTrigger className="h-9 w-44">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__all__">All states</SelectItem>
              {runStates.map((s) => (
                <SelectItem key={s} value={s}>
                  {s}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </label>
        <label className="flex items-center gap-2 text-sm">
          <span className="text-[var(--color-fg-subtle)]">CG ID</span>
          <Input
            value={search.cg_id ?? ""}
            onChange={(e) =>
              void navigate({
                search: {
                  state: search.state,
                  cg_id: e.target.value || undefined,
                  entry_id: search.entry_id,
                  cursor: undefined,
                },
              })
            }
            placeholder="cg_…"
            className="h-9 w-56 font-mono"
          />
        </label>
        <label className="flex items-center gap-2 text-sm">
          <span className="text-[var(--color-fg-subtle)]">Entry ID</span>
          <Input
            value={search.entry_id ?? ""}
            onChange={(e) =>
              void navigate({
                search: {
                  state: search.state,
                  cg_id: search.cg_id,
                  entry_id: e.target.value || undefined,
                  cursor: undefined,
                },
              })
            }
            placeholder="entry_…"
            className="h-9 w-56 font-mono"
          />
        </label>
        {(search.state || search.cg_id || search.entry_id || search.cursor) && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() =>
              void navigate({
                search: {
                  state: undefined,
                  cg_id: undefined,
                  entry_id: undefined,
                  cursor: undefined,
                },
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
            <EmptyState message="No runs match the current filters." />
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Run ID</TableHead>
                  <TableHead>State</TableHead>
                  <TableHead>Seed</TableHead>
                  <TableHead>Duration</TableHead>
                  <TableHead>Attempt</TableHead>
                  <TableHead>Updated</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {query.data.items.map((run) => (
                  <RunRow key={run.id} run={run} />
                ))}
              </TableBody>
            </Table>
          )}
          <LoadMore
            nextCursor={query.data.next_cursor}
            onLoadMore={(cursor) => void navigate({ search: (prev) => ({ ...prev, cursor }) })}
            loading={query.isFetching}
          />
          {search.cursor && (
            <div className="text-center">
              <Link
                to="/runs"
                search={{
                  state: search.state,
                  cg_id: search.cg_id,
                  entry_id: search.entry_id,
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
