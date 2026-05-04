import { createFileRoute, Link } from "@tanstack/react-router";

import { ErrorBanner, LoadingBlock } from "@/components/common/page-states";
import { EntryRow } from "@/components/entity/entry-row";
import { StateBadge } from "@/components/entity/state-badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { $api } from "@/lib/api/client";
import { formatDateTime, formatRelative } from "@/lib/format/datetime";
import type { components } from "@/lib/api/generated/api-types";

type CGDetail = components["schemas"]["ComparisonGroupDetailResponse"];
type AgentStatus = components["schemas"]["AgentStatusResponse"];
type EvaluationResult = components["schemas"]["EvaluationResultResponse"];

export const Route = createFileRoute("/comparison-groups/$id")({
  component: CGDetailPage,
});

function CGDetailPage() {
  const { id } = Route.useParams();
  const detail = $api.useQuery("get", "/api/v1/comparison-groups/{cg_id}", {
    params: { path: { cg_id: id } },
  });
  const agentStatus = $api.useQuery("get", "/api/v1/comparison-groups/{cg_id}/agent-status", {
    params: { path: { cg_id: id } },
  });
  const evaluation = $api.useQuery(
    "get",
    "/api/v1/comparison-groups/{cg_id}/evaluation",
    { params: { path: { cg_id: id } } },
    // The evaluation endpoint 404s until evaluation_result.json is produced —
    // 11-api.md §4.4. Don't refetch in the background once it errors.
    { retry: false },
  );

  return (
    <div className="space-y-4">
      <header>
        <Link
          to="/comparison-groups"
          className="text-xs text-[var(--color-fg-subtle)] hover:underline"
        >
          ← All comparison groups
        </Link>
        <h1 className="mt-1 text-2xl font-bold tracking-tight">
          Comparison Group <code className="font-mono text-base">{id}</code>
        </h1>
      </header>

      {detail.isPending && <LoadingBlock />}
      {detail.error && <ErrorBanner error={detail.error} />}

      {detail.data && (
        <>
          <CGHeader cg={detail.data} />
          <EntriesPanel cg={detail.data} />
          <AgentStatusPanel status={agentStatus.data} isPending={agentStatus.isPending} />
          <EvaluationResultPanel
            cg={detail.data}
            result={evaluation.data}
            isPending={evaluation.isPending}
          />
          <ArtifactsHint cgId={detail.data.id} />
        </>
      )}
    </div>
  );
}

function CGHeader({ cg }: { cg: CGDetail }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-center gap-2">
          <CardTitle>Header</CardTitle>
          <StateBadge state={cg.state} />
          {cg.is_terminal && (
            <span className="rounded bg-[var(--color-bg-subtle)] px-1.5 py-0.5 text-xs text-[var(--color-fg-subtle)]">
              terminal
            </span>
          )}
        </div>
        <CardDescription>Pipeline-tracking record + content-addressed hashes.</CardDescription>
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-1 gap-2 text-sm sm:grid-cols-2">
          <KV label="proposal" value={cg.proposal_id} link={`/proposals/${cg.proposal_id}`} />
          <KV label="experiment_definition_id" value={cg.experiment_definition_id} />
          <KV label="factor_model_id" value={cg.factor_model_id ?? "—"} />
          <KV label="experiment_plan_hash" value={shortHash(cg.experiment_plan_hash)} />
          <KV label="harness_contract_hash" value={shortHash(cg.harness_contract_hash)} />
          <KV label="validation_config_hash" value={shortHash(cg.validation_config_hash)} />
          <KV label="code_review_attempt" value={String(cg.code_review_attempt)} />
          <KV label="code_review_result_hash" value={shortHash(cg.code_review_result_hash)} />
          <KV label="created_at" value={formatDateTime(cg.created_at)} />
          <KV
            label="updated_at"
            value={`${formatDateTime(cg.updated_at)} (${formatRelative(cg.updated_at)})`}
          />
          {cg.last_error && (
            <div className="sm:col-span-2">
              <span className="font-mono text-[var(--color-fg-subtle)]">last_error</span>
              <pre className="mt-1 overflow-x-auto rounded bg-[color-mix(in_oklch,var(--color-danger)_10%,transparent)] p-2 font-mono text-xs text-[var(--color-danger)]">
                {cg.last_error}
              </pre>
            </div>
          )}
        </dl>
      </CardContent>
    </Card>
  );
}

function EntriesPanel({ cg }: { cg: CGDetail }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Entries</CardTitle>
        <CardDescription>
          Each row is one entry derived from the experiment definition; runs are aggregated by
          terminal state.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {cg.entries.length === 0 ? (
          <p className="text-sm text-[var(--color-fg-subtle)]">No entries yet.</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Entry ID</TableHead>
                <TableHead>Technique</TableHead>
                <TableHead>Baseline</TableHead>
                <TableHead>State</TableHead>
                <TableHead>Attempt</TableHead>
                <TableHead>Runs</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {cg.entries.map((entry) => (
                <EntryRow key={entry.id} entry={entry} />
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}

function AgentStatusPanel({
  status,
  isPending,
}: {
  status: AgentStatus | undefined;
  isPending: boolean;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Agent status</CardTitle>
        <CardDescription>
          Composite read of the harness builder's status.json plus per-entry agent status.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        {isPending && <LoadingBlock />}
        {status && (
          <>
            <div>
              <h3 className="text-xs font-semibold tracking-wide text-[var(--color-fg-subtle)] uppercase">
                Harness
              </h3>
              {status.harness == null ? (
                <p className="mt-1 text-[var(--color-fg-subtle)]">
                  Harness status.json not yet written.
                </p>
              ) : (
                <dl className="mt-2 grid grid-cols-1 gap-1 font-mono text-xs sm:grid-cols-2">
                  <KV label="state" value={status.harness.state ?? "—"} />
                  <KV
                    label="turn"
                    value={status.harness.turn != null ? String(status.harness.turn) : "—"}
                  />
                  <KV
                    label="cost_usd"
                    value={
                      status.harness.cost_usd != null
                        ? `$${status.harness.cost_usd.toFixed(4)}`
                        : "—"
                    }
                  />
                  <KV
                    label="last_message_at"
                    value={
                      status.harness.last_message_at
                        ? formatRelative(status.harness.last_message_at)
                        : "—"
                    }
                  />
                </dl>
              )}
            </div>
            <div>
              <h3 className="text-xs font-semibold tracking-wide text-[var(--color-fg-subtle)] uppercase">
                Entries
              </h3>
              {Object.keys(status.entries).length === 0 ? (
                <p className="mt-1 text-[var(--color-fg-subtle)]">
                  No per-entry agent statuses yet.
                </p>
              ) : (
                <ul className="mt-2 space-y-1 font-mono text-xs">
                  {Object.entries(status.entries).map(([entryId, value]) => (
                    <li key={entryId}>
                      <code>{entryId}</code> →{" "}
                      <span className="text-[var(--color-fg-subtle)]">
                        {value.status == null
                          ? "(no status.json yet)"
                          : `technique=${value.technique_id ?? "—"}`}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}

function EvaluationResultPanel({
  cg,
  result,
  isPending,
}: {
  cg: CGDetail;
  result: EvaluationResult | undefined;
  isPending: boolean;
}) {
  // The evaluation endpoint returns 404 until the result artifact lands. We
  // only render the panel when the CG state is past evaluating to avoid a
  // confusing "not yet" panel on every fresh CG.
  if (cg.state !== "complete" && cg.state !== "evaluation_failed") return null;
  return (
    <Card>
      <CardHeader>
        <CardTitle>Evaluation result</CardTitle>
        <CardDescription>
          The mechanical evaluator's verdict for this CG (full structure rendered by 4.M6).
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        {isPending && <LoadingBlock />}
        {result && (
          <>
            <div>
              <span className="text-[var(--color-fg-subtle)]">verdict: </span>
              <span className="font-mono font-medium">{result.verdict}</span>
            </div>
            {result.artifact_key && (
              <div>
                <span className="text-[var(--color-fg-subtle)]">artifact: </span>
                <code className="font-mono text-xs">{result.artifact_key}</code>
              </div>
            )}
            <p className="text-xs text-[var(--color-fg-subtle)]">
              Raw metrics + per-entry verdicts shown as JSON until 4.M6's JsonTree lands.
            </p>
            <pre className="max-h-80 overflow-auto rounded bg-[var(--color-bg-subtle)] p-2 font-mono text-xs">
              {JSON.stringify(
                {
                  raw_metrics: result.raw_metrics,
                  per_entry: result.per_entry,
                  contextual_evaluation: result.contextual_evaluation,
                },
                null,
                2,
              )}
            </pre>
          </>
        )}
      </CardContent>
    </Card>
  );
}

function ArtifactsHint({ cgId }: { cgId: string }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Artifacts</CardTitle>
        <CardDescription>
          Per-CG artifacts (harness/, entries/, evaluation_result.json, code-review.json) live under{" "}
          <code className="font-mono">/api/v1/comparison-groups/{cgId}/artifacts/&lt;path&gt;</code>
          . 4.M6 wires the artifact viewer route at{" "}
          <code className="font-mono">/comparison-groups/{cgId}/artifacts/&lt;path&gt;</code>.
        </CardDescription>
      </CardHeader>
    </Card>
  );
}

function shortHash(hash: string | null): string {
  if (!hash) return "—";
  if (hash.length <= 16) return hash;
  return `${hash.slice(0, 12)}…`;
}

function KV({ label, value, link }: { label: string; value: string; link?: string }) {
  return (
    <div className="flex gap-2 font-mono">
      <dt className="min-w-[10rem] text-[var(--color-fg-subtle)]">{label}</dt>
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
