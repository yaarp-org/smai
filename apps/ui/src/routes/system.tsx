import { createFileRoute } from "@tanstack/react-router";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { $api } from "@/lib/api/client";

export const Route = createFileRoute("/system")({
  component: SystemPage,
});

// 4.M2 sample-usage demonstration: round-trips a typed query through the
// generated $api client to prove the codegen pipeline end-to-end. Real page
// content (plugin status, verify, migrate-status per 13-frontend.md §11.2)
// lands in 4.M3.
function SystemPage() {
  const { data, isPending, error } = $api.useQuery("get", "/api/v1/system/version");

  return (
    <Card>
      <CardHeader>
        <CardTitle>System</CardTitle>
        <CardDescription>
          Plugin status, version, verify, migrate-status (per 13-frontend.md §11.2).
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        {isPending && <p className="text-[var(--color-fg-subtle)]">Loading version...</p>}
        {error && (
          <p className="text-[var(--color-danger)]">Failed to fetch version: {String(error)}</p>
        )}
        {data && (
          <dl className="font-mono">
            <div className="flex gap-2">
              <dt className="text-[var(--color-fg-subtle)]">smai-cli</dt>
              <dd>{data.smai_cli}</dd>
            </div>
            <div className="flex gap-2">
              <dt className="text-[var(--color-fg-subtle)]">smai-core</dt>
              <dd>{data.smai_core}</dd>
            </div>
            <div className="flex gap-2">
              <dt className="text-[var(--color-fg-subtle)]">smai-api-spec</dt>
              <dd>{data.smai_api_spec}</dd>
            </div>
          </dl>
        )}
        <p className="text-[var(--color-fg-subtle)]">
          Real content lands in <span className="font-mono">Task 4.M3</span>.
        </p>
      </CardContent>
    </Card>
  );
}
