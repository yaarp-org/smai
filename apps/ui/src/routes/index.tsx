import { useQuery } from "@tanstack/react-query";
import { createFileRoute, Link } from "@tanstack/react-router";

import { ErrorBanner, LoadingBlock } from "@/components/common/page-states";
import { StateBadge } from "@/components/entity/state-badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { $api } from "@/lib/api/client";
import {
  SSE_STATUS_QUERY_KEY,
  WORKER_HEARTBEAT_QUERY_KEY,
  type SseConnectionStatus,
  type WorkerHeartbeatEvent,
} from "@/lib/events/sse";
import { formatRelative } from "@/lib/format/datetime";
import type { components } from "@/lib/api/generated/api-types";

export const Route = createFileRoute("/")({
  component: DashboardPage,
});

type RecentActivityItem = components["schemas"]["RecentActivityItem"];

function DashboardPage() {
  const { data, isPending, error } = $api.useQuery("get", "/api/v1/system/dashboard");

  return (
    <div className="space-y-6">
      <header className="space-y-1">
        <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-sm text-[var(--color-fg-subtle)]">
          What's running right now and what changed recently.
        </p>
      </header>

      {isPending && <LoadingBlock />}
      {error && <ErrorBanner error={error} />}

      {data && (
        <>
          <SummaryCounts counts={data.counts} />
          <RecentActivityFeed items={data.recent_activity} />
        </>
      )}

      <WorkerHeartbeatFooter />
    </div>
  );
}

function SummaryCounts({ counts }: { counts: components["schemas"]["SummaryCounts"] }) {
  const tiles = [
    { label: "Proposals", value: counts.proposals_in_flight, to: "/proposals" as const },
    { label: "Papers", value: counts.papers_in_flight, to: "/papers" as const },
    { label: "Comparison Groups", value: counts.cgs_in_flight, to: "/comparison-groups" as const },
    { label: "Runs", value: counts.runs_in_flight, to: "/runs" as const },
  ];
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {tiles.map((tile) => (
        <Link
          key={tile.label}
          to={tile.to}
          className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] p-4 transition-colors hover:bg-[var(--color-bg-subtle)]"
        >
          <div className="text-xs font-medium text-[var(--color-fg-subtle)]">{tile.label}</div>
          <div className="mt-1 text-2xl font-bold tabular-nums">{tile.value}</div>
          <div className="text-xs text-[var(--color-fg-subtle)]">in flight</div>
        </Link>
      ))}
    </div>
  );
}

function RecentActivityFeed({ items }: { items: RecentActivityItem[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Recent activity</CardTitle>
        <CardDescription>
          The latest state changes across proposals, papers, comparison groups, entries, and runs.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {items.length === 0 ? (
          <p className="text-sm text-[var(--color-fg-subtle)]">No recent activity.</p>
        ) : (
          <ul className="divide-y divide-[var(--color-border)]">
            {items.map((item, idx) => (
              <ActivityRow key={`${item.kind}:${item.id}:${item.ts}:${idx}`} item={item} />
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

function ActivityRow({ item }: { item: RecentActivityItem }) {
  const link = activityLinkFor(item);
  const inner = (
    <div className="flex flex-wrap items-center gap-2 py-2 text-sm">
      <span className="rounded bg-[var(--color-bg-subtle)] px-1.5 py-0.5 font-mono text-xs text-[var(--color-fg-subtle)]">
        {item.kind}
      </span>
      <code className="font-mono text-xs">{shortId(item.id)}</code>
      <StateBadge state={item.state} />
      <span className="ml-auto text-xs text-[var(--color-fg-subtle)]" title={item.ts}>
        {formatRelative(item.ts)}
      </span>
    </div>
  );
  if (!link) return <li>{inner}</li>;
  return (
    <li>
      <Link to={link} className="block hover:bg-[var(--color-bg-subtle)]">
        {inner}
      </Link>
    </li>
  );
}

function activityLinkFor(item: RecentActivityItem): string | null {
  // Entries and runs both take the user to the parent CG detail page since
  // entry detail needs both cg_id and entry_id and the recent-activity feed
  // exposes only the entity id. Runs have a flat /runs/$id route. Proposals,
  // papers, and CGs land on their own detail.
  switch (item.kind) {
    case "proposal":
      return `/proposals/${item.id}`;
    case "paper":
      return `/papers/${item.id}`;
    case "comparison_group":
      return `/comparison-groups/${item.id}`;
    case "run":
      return `/runs/${item.id}`;
    case "entry":
      return null;
    default:
      return null;
  }
}

function shortId(id: string): string {
  if (id.length <= 16) return id;
  return `${id.slice(0, 8)}…${id.slice(-4)}`;
}

function WorkerHeartbeatFooter() {
  // Both the heartbeat payload and the SSE connection status are pushed into
  // the TanStack Query cache by `lib/events/sse.ts` (setQueryData). useQuery
  // with `enabled: false` reads the cache and re-renders on every push; the
  // queryFn is a placeholder that never runs.
  const { data: heartbeat } = useQuery<WorkerHeartbeatEvent | undefined>({
    queryKey: WORKER_HEARTBEAT_QUERY_KEY,
    queryFn: () => Promise.resolve(undefined),
    enabled: false,
    staleTime: Infinity,
  });
  const { data: status } = useQuery<SseConnectionStatus | undefined>({
    queryKey: SSE_STATUS_QUERY_KEY,
    queryFn: () => Promise.resolve(undefined),
    enabled: false,
    staleTime: Infinity,
  });

  return (
    <footer className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-[var(--color-border)] pt-3 text-xs text-[var(--color-fg-subtle)]">
      <span className="inline-flex items-center gap-1.5">
        <ConnectionDot status={status} />
        <span className="font-mono">SSE:</span>
        <span>{status ?? "connecting"}</span>
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className="font-mono">worker:</span>
        {heartbeat ? (
          <span className="tabular-nums">
            cycle {heartbeat.cycle_id} · {heartbeat.cycles_processed} processed · last beat{" "}
            <span title={heartbeat.ts}>{formatRelative(heartbeat.ts)}</span>
          </span>
        ) : (
          <span className="italic">(awaiting worker)</span>
        )}
      </span>
    </footer>
  );
}

function ConnectionDot({ status }: { status: SseConnectionStatus | undefined }) {
  // green=open, amber=connecting (the initial state and any transient
  // reconnect), red=closed (server gave up or bearer-mode deferred). Matches
  // the EventSource readyState mapping in `lib/events/sse.ts`.
  const tone =
    status === "open" ? "bg-green-500" : status === "closed" ? "bg-red-500" : "bg-amber-500";
  const label =
    status === "open"
      ? "Live updates connected"
      : status === "closed"
        ? "Live updates disconnected"
        : "Live updates connecting";
  return (
    <span
      aria-label={label}
      title={label}
      className={`inline-block h-2 w-2 rounded-full ${tone}`}
    />
  );
}
