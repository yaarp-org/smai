import { createFileRoute, Link } from "@tanstack/react-router";
import { z } from "zod";

import { LoadMore } from "@/components/common/load-more";
import { EmptyState, ErrorBanner, LoadingBlock } from "@/components/common/page-states";
import { PaperCard } from "@/components/entity/paper-card";
import { Button, buttonVariants } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { $api } from "@/lib/api/client";

const paperStates = [
  "submitted",
  "fetching",
  "screening",
  "planning",
  "registered",
  "rejected",
  "failed",
  "partial",
] as const;
type PaperStateLiteral = (typeof paperStates)[number];

const paperListSearchSchema = z.object({
  state: z.enum(paperStates).optional(),
  cursor: z.string().optional(),
});

export const Route = createFileRoute("/papers/")({
  validateSearch: paperListSearchSchema,
  component: PapersListPage,
});

function PapersListPage() {
  const search = Route.useSearch();
  const navigate = Route.useNavigate();

  const query = $api.useQuery("get", "/api/v1/papers", {
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
          <h1 className="text-2xl font-bold tracking-tight">Papers</h1>
          <p className="text-sm text-[var(--color-fg-subtle)]">
            Pull an arXiv paper's techniques into the registry so a proposal can reproduce its
            results.
          </p>
        </div>
        <Link to="/papers/new" className={buttonVariants()}>
          Ingest by arXiv ID
        </Link>
      </header>

      <div className="flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-2 text-sm">
          <span className="text-[var(--color-fg-subtle)]">State</span>
          <Select
            value={search.state ?? "__all__"}
            onValueChange={(value) => {
              const next: PaperStateLiteral | undefined =
                value === "__all__" ? undefined : (value as PaperStateLiteral);
              void navigate({ search: { state: next, cursor: undefined } });
            }}
          >
            <SelectTrigger className="h-9 w-44">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__all__">All states</SelectItem>
              {paperStates.map((s) => (
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
            <EmptyState message="No papers match the current filters." />
          ) : (
            <ul className="space-y-2">
              {query.data.items.map((paper) => (
                <li key={paper.arxiv_id}>
                  <PaperCard paper={paper} />
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
                to="/papers"
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
