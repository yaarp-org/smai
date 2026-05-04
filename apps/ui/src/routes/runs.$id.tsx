import { createFileRoute, Link } from "@tanstack/react-router";

import { ErrorBanner, LoadingBlock } from "@/components/common/page-states";
import { StateBadge } from "@/components/entity/state-badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { $api } from "@/lib/api/client";
import { formatDateTime, formatRelative } from "@/lib/format/datetime";
import { formatDuration } from "@/lib/format/duration";
import type { components } from "@/lib/api/generated/api-types";

type RunDetail = components["schemas"]["RunDetailResponse"];

export const Route = createFileRoute("/runs/$id")({
  component: RunDetailPage,
});

function RunDetailPage() {
  const { id } = Route.useParams();
  const query = $api.useQuery("get", "/api/v1/runs/{run_id}", {
    params: { path: { run_id: id } },
  });

  return (
    <div className="space-y-4">
      <header>
        <Link to="/runs" className="text-xs text-[var(--color-fg-subtle)] hover:underline">
          ← All runs
        </Link>
        <h1 className="mt-1 text-2xl font-bold tracking-tight">
          Run <code className="font-mono text-base">{id}</code>
        </h1>
      </header>

      {query.isPending && <LoadingBlock />}
      {query.error && <ErrorBanner error={query.error} />}

      {query.data && (
        <>
          <RunHeader run={query.data} />
          <MetricsPanel run={query.data} />
        </>
      )}
    </div>
  );
}

function RunHeader({ run }: { run: RunDetail }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-center gap-2">
          <CardTitle>Header</CardTitle>
          <StateBadge state={run.state} />
        </div>
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-1 gap-2 font-mono text-sm sm:grid-cols-2">
          <KV label="cg_id" value={run.cg_id} link={`/comparison-groups/${run.cg_id}`} />
          <KV
            label="entry_id"
            value={run.entry_id}
            link={`/comparison-groups/${run.cg_id}/entries/${run.entry_id}`}
          />
          <KV label="seed" value={String(run.seed)} />
          <KV label="run_attempt" value={String(run.run_attempt)} />
          <KV label="duration" value={formatDuration(run.duration_seconds)} />
          <KV label="started_at" value={run.started_at ? formatDateTime(run.started_at) : "—"} />
          <KV
            label="completed_at"
            value={run.completed_at ? formatDateTime(run.completed_at) : "—"}
          />
          <KV label="created_at" value={formatDateTime(run.created_at)} />
          <KV
            label="updated_at"
            value={`${formatDateTime(run.updated_at)} (${formatRelative(run.updated_at)})`}
          />
          {run.failure_reason && (
            <div className="sm:col-span-2">
              <span className="text-[var(--color-fg-subtle)]">failure_reason</span>
              <pre className="mt-1 overflow-x-auto rounded bg-[color-mix(in_oklch,var(--color-danger)_10%,transparent)] p-2 font-mono text-xs text-[var(--color-danger)]">
                {run.failure_reason}
              </pre>
            </div>
          )}
          {run.last_error && (
            <div className="sm:col-span-2">
              <span className="text-[var(--color-fg-subtle)]">last_error</span>
              <pre className="mt-1 overflow-x-auto rounded bg-[color-mix(in_oklch,var(--color-danger)_10%,transparent)] p-2 font-mono text-xs text-[var(--color-danger)]">
                {run.last_error}
              </pre>
            </div>
          )}
        </dl>
      </CardContent>
    </Card>
  );
}

function MetricsPanel({ run }: { run: RunDetail }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Metrics</CardTitle>
        <CardDescription>
          Raw metrics produced by the run (rendered as JSON until 4.M6's JsonTree lands).
        </CardDescription>
      </CardHeader>
      <CardContent>
        {run.raw_metrics_artifact_key ? (
          <p className="font-mono text-sm">
            <span className="text-[var(--color-fg-subtle)]">artifact: </span>
            {run.raw_metrics_artifact_key}
          </p>
        ) : (
          <p className="text-sm text-[var(--color-fg-subtle)]">No metrics artifact recorded yet.</p>
        )}
      </CardContent>
    </Card>
  );
}

function KV({ label, value, link }: { label: string; value: string; link?: string }) {
  return (
    <div className="flex gap-2">
      <dt className="min-w-[8rem] text-[var(--color-fg-subtle)]">{label}</dt>
      <dd>
        {link ? (
          <Link to={link} className="text-[var(--color-accent)] hover:underline">
            {value}
          </Link>
        ) : (
          value
        )}
      </dd>
    </div>
  );
}
