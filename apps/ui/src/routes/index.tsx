import { createFileRoute, Link } from "@tanstack/react-router";

import { ErrorBanner, LoadingBlock } from "@/components/common/page-states";
import { StateBadge } from "@/components/entity/state-badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { $api } from "@/lib/api/client";
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
          In-flight entity counts and recent state transitions across the deployment.
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
          Most recent state transitions across all kinds (proposals, papers, comparison groups,
          entries, runs).
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

// 4.M5 will populate the worker_heartbeat query data via SSE; M3 reads it via
// queryClient.getQueryData(["worker_heartbeat"]). Until then the tile shows
// "(awaiting worker)". Wired now so M5 just needs to push values.
function WorkerHeartbeatFooter() {
  // Leaving as static placeholder — TanStack Query's queryClient is not exposed
  // to this component yet (no useQueryClient() reads needed in M3), and per the
  // brief we keep this empty until M5 wires the SSE channel.
  return (
    <footer className="border-t border-[var(--color-border)] pt-3 text-xs text-[var(--color-fg-subtle)]">
      <span className="font-mono">worker:</span>{" "}
      <span className="italic">(awaiting worker — SSE wires in 4.M5)</span>
    </footer>
  );
}
