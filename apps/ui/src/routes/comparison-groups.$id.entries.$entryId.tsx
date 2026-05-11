import { createFileRoute, Link } from "@tanstack/react-router";

import { ErrorBanner, LoadingBlock } from "@/components/common/page-states";
import { RunRow } from "@/components/entity/run-row";
import { StateBadge } from "@/components/entity/state-badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { $api } from "@/lib/api/client";
import { formatDateTime, formatRelative } from "@/lib/format/datetime";
import type { components } from "@/lib/api/generated/api-types";

type EntryDetail = components["schemas"]["EntryDetailResponse"];

export const Route = createFileRoute("/comparison-groups/$id/entries/$entryId")({
  component: EntryDetailPage,
});

function EntryDetailPage() {
  const { id, entryId } = Route.useParams();
  const query = $api.useQuery("get", "/api/v1/comparison-groups/{cg_id}/entries/{entry_id}", {
    params: { path: { cg_id: id, entry_id: entryId } },
  });

  return (
    <div className="space-y-4">
      <header>
        <Link
          to="/comparison-groups/$id"
          params={{ id }}
          className="text-xs text-[var(--color-fg-subtle)] hover:underline"
        >
          ← Back to CG
        </Link>
        <h1 className="mt-1 text-2xl font-bold tracking-tight">
          Entry <code className="font-mono text-base">{entryId}</code>
        </h1>
      </header>

      {query.isPending && <LoadingBlock />}
      {query.error && <ErrorBanner error={query.error} />}

      {query.data && (
        <>
          <EntryHeader entry={query.data} />
          <TechniqueContractPanel entry={query.data} />
          <CodePanel entry={query.data} />
          <RunsPanel entry={query.data} />
          <DiffToBaselinePlaceholder />
        </>
      )}
    </div>
  );
}

function EntryHeader({ entry }: { entry: EntryDetail }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-center gap-2">
          <CardTitle>Header</CardTitle>
          <StateBadge state={entry.state} />
          {entry.is_baseline && (
            <span className="rounded bg-[var(--color-bg-subtle)] px-1.5 py-0.5 text-xs">
              baseline
            </span>
          )}
        </div>
        <CardDescription>
          One entry of the comparison: either the baseline or one technique variant.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-1 gap-2 font-mono text-sm sm:grid-cols-2">
          <KV label="cg_id" value={entry.cg_id} />
          <KV label="technique_id" value={entry.technique_id ?? "—"} />
          <KV label="implementation_attempt" value={String(entry.implementation_attempt)} />
          <KV label="technique_contract_hash" value={shortHash(entry.technique_contract_hash)} />
          <KV
            label="harness_api_manifest_hash"
            value={shortHash(entry.harness_api_manifest_hash)}
          />
          <KV label="created_at" value={formatDateTime(entry.created_at)} />
          <KV
            label="updated_at"
            value={`${formatDateTime(entry.updated_at)} (${formatRelative(entry.updated_at)})`}
          />
          {entry.last_error && (
            <div className="sm:col-span-2">
              <span className="text-[var(--color-fg-subtle)]">last_error</span>
              <pre className="mt-1 overflow-x-auto rounded bg-[color-mix(in_oklch,var(--color-danger)_10%,transparent)] p-2 font-mono text-xs text-[var(--color-danger)]">
                {entry.last_error}
              </pre>
            </div>
          )}
        </dl>
      </CardContent>
    </Card>
  );
}

function TechniqueContractPanel({ entry }: { entry: EntryDetail }) {
  if (!entry.technique_contract_hash) return null;
  return (
    <Card>
      <CardHeader>
        <CardTitle>Technique contract</CardTitle>
        <CardDescription>
          The contract this technique was compiled against. Locked at compile time and
          content-addressed so the same input always produces the same hash.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <p className="font-mono text-sm">
          <span className="text-[var(--color-fg-subtle)]">hash: </span>
          {entry.technique_contract_hash}
        </p>
        <p className="mt-2 text-xs text-[var(--color-fg-subtle)]">
          The contract JSON lives at{" "}
          <code className="font-mono">
            /comparison-groups/{entry.cg_id}/artifacts/entries/{entry.id}/technique_contract.json
          </code>
          . Inline viewing lands later.
        </p>
      </CardContent>
    </Card>
  );
}

function CodePanel({ entry }: { entry: EntryDetail }) {
  // Per the brief: render filenames as monospace. M6 wires the artifact
  // viewer route. The list of files is not currently exposed by a dedicated
  // contract endpoint — we direct readers to the artifact list URL.
  return (
    <Card>
      <CardHeader>
        <CardTitle>Code</CardTitle>
        <CardDescription>
          The source the technique-implementer agent wrote for this entry.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <p className="text-sm">
          Source files live under{" "}
          <code className="font-mono text-xs">
            /comparison-groups/{entry.cg_id}/artifacts/entries/{entry.id}/code/
          </code>
          .
        </p>
        <p className="mt-2 text-xs text-[var(--color-fg-subtle)]">
          Syntax-highlighted inline viewing lands later.
        </p>
      </CardContent>
    </Card>
  );
}

function RunsPanel({ entry }: { entry: EntryDetail }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Runs</CardTitle>
        <CardDescription>One row per random seed this entry was run at.</CardDescription>
      </CardHeader>
      <CardContent>
        {entry.runs.length === 0 ? (
          <p className="text-sm text-[var(--color-fg-subtle)]">
            No runs yet. They start once this entry's code has been implemented and dispatched.
          </p>
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
              {entry.runs.map((run) => (
                <RunRow key={run.id} run={run} />
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}

function DiffToBaselinePlaceholder() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Diff to baseline</CardTitle>
        <CardDescription>Side-by-side diff of technique source vs baseline source.</CardDescription>
      </CardHeader>
      <CardContent>
        <Tooltip>
          <TooltipTrigger asChild>
            <span>
              <Button disabled variant="outline">
                Compare to baseline
              </Button>
            </span>
          </TooltipTrigger>
          <TooltipContent>Side-by-side diff lands in a follow-up.</TooltipContent>
        </Tooltip>
      </CardContent>
    </Card>
  );
}

function shortHash(hash: string | null): string {
  if (!hash) return "—";
  if (hash.length <= 16) return hash;
  return `${hash.slice(0, 12)}…`;
}

function KV({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex gap-2">
      <dt className="min-w-[10rem] text-[var(--color-fg-subtle)]">{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}
