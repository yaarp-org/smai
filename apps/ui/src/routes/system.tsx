import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";

import { ErrorBanner, LoadingBlock } from "@/components/common/page-states";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { fetchClient, $api } from "@/lib/api/client";
import type { components } from "@/lib/api/generated/api-types";

export const Route = createFileRoute("/system")({
  component: SystemPage,
});

type SystemVerifyResponse = components["schemas"]["SystemVerifyResponse"];
type PluginVerifyResult = components["schemas"]["PluginVerifyResult"];

interface ApiErrorEnvelope {
  error?: { message?: string };
}

function extractMessage(err: unknown): string {
  if (err && typeof err === "object") {
    const env = err as ApiErrorEnvelope;
    if (env.error?.message) return env.error.message;
    if (err instanceof Error) return err.message;
  }
  return String(err);
}

function SystemPage() {
  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-2xl font-bold tracking-tight">System</h1>
        <p className="text-sm text-[var(--color-fg-subtle)]">
          Versions, plugins, database migration status, and the verify pre-flight that pings every
          configured plugin.
        </p>
      </header>
      <Tabs defaultValue="version" className="w-full">
        <TabsList>
          <TabsTrigger value="version">Version</TabsTrigger>
          <TabsTrigger value="plugins">Plugins</TabsTrigger>
          <TabsTrigger value="migrate">Migrate</TabsTrigger>
          <TabsTrigger value="verify">Verify</TabsTrigger>
        </TabsList>
        <TabsContent value="version">
          <VersionTab />
        </TabsContent>
        <TabsContent value="plugins">
          <PluginsTab />
        </TabsContent>
        <TabsContent value="migrate">
          <MigrateStatusTab />
        </TabsContent>
        <TabsContent value="verify">
          <VerifyTab />
        </TabsContent>
      </Tabs>
    </div>
  );
}

function VersionTab() {
  const { data, isPending, error } = $api.useQuery("get", "/api/v1/system/version");
  return (
    <Card>
      <CardHeader>
        <CardTitle>Versions</CardTitle>
        <CardDescription>
          Versions of the core SMAI packages and every plugin distribution currently loaded.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        {isPending && <LoadingBlock />}
        {error && <ErrorBanner error={error} />}
        {data && (
          <>
            <dl className="grid grid-cols-1 gap-1 font-mono sm:grid-cols-2">
              <KeyVal label="smai-cli" value={data.smai_cli} />
              <KeyVal label="smai-core" value={data.smai_core} />
              <KeyVal label="smai-api-spec" value={data.smai_api_spec} />
            </dl>
            <div>
              <h3 className="mb-2 text-xs font-semibold tracking-wide text-[var(--color-fg-subtle)] uppercase">
                Plugins
              </h3>
              {Object.keys(data.plugins).length === 0 ? (
                <p className="text-[var(--color-fg-subtle)]">No plugin distributions loaded.</p>
              ) : (
                <dl className="grid grid-cols-1 gap-1 font-mono sm:grid-cols-2">
                  {Object.entries(data.plugins).map(([name, version]) => (
                    <KeyVal key={name} label={name} value={version} />
                  ))}
                </dl>
              )}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}

function KeyVal({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex gap-2">
      <dt className="min-w-[10rem] text-[var(--color-fg-subtle)]">{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function PluginsTab() {
  const { data, isPending, error } = $api.useQuery("get", "/api/v1/system/plugins");
  return (
    <Card>
      <CardHeader>
        <CardTitle>Discovered plugins</CardTitle>
        <CardDescription>
          Every plugin pip can find, grouped by what it plugs into. The "selected" one is the plugin
          your smai.yaml is using right now.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {isPending && <LoadingBlock />}
        {error && <ErrorBanner error={error} />}
        {data && (
          <>
            <PluginGroup title="LLM providers" rows={data.llm_providers} />
            <PluginGroup title="Metadata stores" rows={data.metadata_stores} />
            <PluginGroup title="Artifact stores" rows={data.artifact_stores} />
            <PluginGroup title="Computes" rows={data.computes} />
          </>
        )}
      </CardContent>
    </Card>
  );
}

function PluginGroup({
  title,
  rows,
}: {
  title: string;
  rows: components["schemas"]["PluginInfo"][];
}) {
  return (
    <div>
      <h3 className="mb-2 text-xs font-semibold tracking-wide text-[var(--color-fg-subtle)] uppercase">
        {title}
      </h3>
      {rows.length === 0 ? (
        <p className="text-sm text-[var(--color-fg-subtle)]">None discovered.</p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead>Distribution</TableHead>
              <TableHead>Version</TableHead>
              <TableHead>Selected</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row) => (
              <TableRow key={`${row.distribution}::${row.name}`}>
                <TableCell className="font-mono">{row.name}</TableCell>
                <TableCell className="font-mono">{row.distribution}</TableCell>
                <TableCell className="font-mono">{row.version}</TableCell>
                <TableCell>
                  {row.selected ? (
                    <span className="font-medium text-[var(--color-success)]">selected</span>
                  ) : (
                    <span className="text-[var(--color-fg-subtle)]">—</span>
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </div>
  );
}

function MigrateStatusTab() {
  const { data, isPending, error } = $api.useQuery("get", "/api/v1/system/migrate-status");
  return (
    <Card>
      <CardHeader>
        <CardTitle>Database migration status</CardTitle>
        <CardDescription>
          Whether the database schema matches the version SMAI expects. Equivalent to running
          <code className="mx-1 font-mono">smai migrate --check</code> on the CLI.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        {isPending && <LoadingBlock />}
        {error && <ErrorBanner error={error} />}
        {data && (
          <>
            <div className="flex items-center gap-2">
              {data.at_head ? (
                <span className="inline-flex items-center gap-2 rounded-md bg-[color-mix(in_oklch,var(--color-success)_15%,transparent)] px-2 py-1 text-xs font-medium text-[var(--color-success)] ring-1 ring-[color-mix(in_oklch,var(--color-success)_30%,transparent)] ring-inset">
                  ✓ at head
                </span>
              ) : (
                <span className="inline-flex items-center gap-2 rounded-md bg-[color-mix(in_oklch,var(--color-danger)_15%,transparent)] px-2 py-1 text-xs font-medium text-[var(--color-danger)] ring-1 ring-[color-mix(in_oklch,var(--color-danger)_30%,transparent)] ring-inset">
                  ✗ schema drift
                </span>
              )}
            </div>
            <dl className="grid grid-cols-1 gap-1 font-mono">
              <KeyVal label="head_revision" value={data.head_revision} />
              <KeyVal label="current" value={data.current ?? "(none — fresh database)"} />
            </dl>
          </>
        )}
      </CardContent>
    </Card>
  );
}

// 11-api.md §5.2.5 / §4.6: the verify endpoint runs four real plugin probes
// and ALWAYS returns 200 with body. It costs LLM tokens (the Bedrock Converse
// LlmProvider issues a tiny prompt). The button is wired here in M3 since the
// task brief explicitly calls it out as an idempotent system action, not a
// state-machine mutation that belongs to M4.
function VerifyTab() {
  const [data, setData] = useState<SystemVerifyResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  async function runVerify() {
    setRunning(true);
    setError(null);
    try {
      const result = await fetchClient.POST("/api/v1/system/verify", {
        body: { plugins: null },
      });
      if (result.error) {
        setError(extractMessage(result.error));
      } else if (result.data) {
        setData(result.data);
      }
    } catch (e) {
      setError(extractMessage(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Verify</CardTitle>
        <CardDescription>
          Pings every configured plugin to check the credentials and connectivity are working. The
          LLM probe issues a tiny prompt, so running this <strong>costs real LLM tokens</strong>.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Button onClick={runVerify} disabled={running}>
          {running ? "Verifying…" : "Run verify"}
        </Button>
        {error && <ErrorBanner error={error} />}
        {data && (
          <div className="space-y-3">
            <div className="text-sm">
              Overall:{" "}
              {data.overall_ok ? (
                <span className="font-medium text-[var(--color-success)]">PASS</span>
              ) : (
                <span className="font-medium text-[var(--color-danger)]">FAIL</span>
              )}
            </div>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Plugin</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Latency</TableHead>
                  <TableHead>Reason</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                <VerifyRow label="llm_provider" result={data.llm_provider} />
                <VerifyRow label="metadata_store" result={data.metadata_store} />
                <VerifyRow label="artifact_store" result={data.artifact_store} />
                <VerifyRow label="compute" result={data.compute} />
              </TableBody>
            </Table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function VerifyRow({ label, result }: { label: string; result: PluginVerifyResult }) {
  return (
    <TableRow>
      <TableCell className="font-mono">{label}</TableCell>
      <TableCell>
        {result.ok ? (
          <span className="font-medium text-[var(--color-success)]">PASS</span>
        ) : (
          <span className="font-medium text-[var(--color-danger)]">FAIL</span>
        )}
      </TableCell>
      <TableCell className="font-mono">
        {result.latency_ms == null ? "—" : `${result.latency_ms.toFixed(0)}ms`}
      </TableCell>
      <TableCell className="text-[var(--color-fg-subtle)]">{result.reason}</TableCell>
    </TableRow>
  );
}
