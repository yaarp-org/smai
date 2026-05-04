import { Link, createFileRoute } from "@tanstack/react-router";

import { ArtifactFrame } from "@/components/viewers";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export const Route = createFileRoute("/comparison-groups/$id/artifacts/$")({
  component: ArtifactPage,
});

function ArtifactPage() {
  const { id, _splat: rawSplat } = Route.useParams();
  const path = rawSplat ?? "";

  return (
    <div className="space-y-4">
      <header>
        <Link
          to="/comparison-groups/$id"
          params={{ id }}
          className="text-xs text-[var(--color-fg-subtle)] hover:underline"
        >
          ← Back to comparison group
        </Link>
        <h1 className="mt-1 text-2xl font-bold tracking-tight">Artifact</h1>
        <Breadcrumbs cgId={id} path={path} />
      </header>

      {path === "" ? (
        <Card>
          <CardHeader>
            <CardTitle>No artifact selected</CardTitle>
            <CardDescription>
              Append a path under{" "}
              <code className="font-mono">/comparison-groups/{id}/artifacts/</code> to view a
              specific file (for example <code className="font-mono">harness/contract.json</code>).
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-[var(--color-fg-subtle)]">
              An artifact-listing browser lands as a follow-up; this route currently expects a
              direct path link.
            </p>
          </CardContent>
        </Card>
      ) : (
        <ArtifactFrame cgId={id} path={path} />
      )}
    </div>
  );
}

interface BreadcrumbsProps {
  cgId: string;
  path: string;
}

// Each path segment links back to the CG detail page with the prefix encoded
// into a query param the listing UI (future) can consume. Today the parent
// segments are not deep-linkable to a list view, so they fall back to the CG
// detail page link rather than a broken artifact-prefix route.
function Breadcrumbs({ cgId, path }: BreadcrumbsProps) {
  const segments = path === "" ? [] : path.split("/");
  return (
    <nav
      aria-label="Artifact path"
      className="mt-2 flex flex-wrap items-center gap-1 font-mono text-xs text-[var(--color-fg-subtle)]"
    >
      <Link to="/comparison-groups/$id" params={{ id: cgId }} className="hover:underline">
        {cgId}
      </Link>
      <span>/</span>
      <span className="text-[var(--color-fg-subtle)]">artifacts</span>
      {segments.map((segment, idx) => {
        const isLast = idx === segments.length - 1;
        return (
          <span key={`${segment}-${idx}`} className="flex items-center gap-1">
            <span>/</span>
            {isLast ? (
              <span className="text-[var(--color-fg)]">{segment}</span>
            ) : (
              <span>{segment}</span>
            )}
          </span>
        );
      })}
    </nav>
  );
}
