import { useQueryClient } from "@tanstack/react-query";
import { createFileRoute, Link } from "@tanstack/react-router";
import { toast } from "sonner";

import { ErrorBanner, LoadingBlock } from "@/components/common/page-states";
import { StateBadge } from "@/components/entity/state-badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { $api } from "@/lib/api/client";
import { errorMessage, invalidateAfterPaperPromote } from "@/lib/api/mutation-helpers";
import { formatDateTime, formatRelative } from "@/lib/format/datetime";
import type { components } from "@/lib/api/generated/api-types";

type PaperDetail = components["schemas"]["PaperDetailResponse"];
type TechniqueRef = components["schemas"]["TechniqueRefSummary"];

export const Route = createFileRoute("/papers/$arxivId")({
  component: PaperDetailPage,
});

function PaperDetailPage() {
  const { arxivId } = Route.useParams();
  const query = $api.useQuery("get", "/api/v1/papers/{arxiv_id}", {
    params: { path: { arxiv_id: arxivId } },
  });

  return (
    <div className="space-y-4">
      <header>
        <Link to="/papers" className="text-xs text-[var(--color-fg-subtle)] hover:underline">
          ← All papers
        </Link>
        <h1 className="mt-1 text-2xl font-bold tracking-tight">
          arXiv:<code className="ml-1 font-mono text-base">{arxivId}</code>
        </h1>
      </header>

      {query.isPending && <LoadingBlock />}
      {query.error && <ErrorBanner error={query.error} />}

      {query.data && (
        <>
          <PaperHeader paper={query.data} />
          <TechniqueRefsPanel paper={query.data} />
          <PromotePartialPanel paper={query.data} />
          <ArtifactsPanel paper={query.data} />
        </>
      )}
    </div>
  );
}

function PaperHeader({ paper }: { paper: PaperDetail }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-center gap-2">
          <CardTitle>{paper.title ?? paper.arxiv_id}</CardTitle>
          <StateBadge state={paper.state} />
        </div>
        {paper.authors.length > 0 && (
          <CardDescription>
            {paper.authors.length > 6
              ? `${paper.authors.slice(0, 6).join(", ")}, +${paper.authors.length - 6}`
              : paper.authors.join(", ")}
          </CardDescription>
        )}
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        {paper.abstract && (
          <p className="leading-relaxed text-[var(--color-fg-subtle)]">{paper.abstract}</p>
        )}
        <dl className="grid grid-cols-1 gap-2 text-sm sm:grid-cols-2">
          <KV
            label="published_date"
            value={paper.published_date ? formatDateTime(paper.published_date) : "—"}
          />
          <KV
            label="categories"
            value={paper.categories.length ? paper.categories.join(", ") : "—"}
          />
          <KV label="created_at" value={formatDateTime(paper.created_at)} />
          <KV
            label="updated_at"
            value={`${formatDateTime(paper.updated_at)} (${formatRelative(paper.updated_at)})`}
          />
          <KV label="planning_attempt" value={String(paper.planning_attempt)} />
          <KV label="screening_attempt" value={String(paper.screening_attempt)} />
          <KV label="registration_attempt" value={String(paper.registration_attempt)} />
          <KV label="screen_decision" value={paper.screen_result_decision ?? "—"} />
          {paper.screen_result_reason && (
            <div className="sm:col-span-2">
              <span className="font-mono text-[var(--color-fg-subtle)]">screen_result_reason</span>
              <p className="mt-1 text-sm">{paper.screen_result_reason}</p>
            </div>
          )}
          {paper.last_error && (
            <div className="sm:col-span-2">
              <span className="font-mono text-[var(--color-fg-subtle)]">last_error</span>
              <pre className="mt-1 overflow-x-auto rounded bg-[color-mix(in_oklch,var(--color-danger)_10%,transparent)] p-2 font-mono text-xs text-[var(--color-danger)]">
                {paper.last_error}
              </pre>
            </div>
          )}
        </dl>
      </CardContent>
    </Card>
  );
}

function TechniqueRefsPanel({ paper }: { paper: PaperDetail }) {
  if (paper.state !== "registered") return null;
  return (
    <Card>
      <CardHeader>
        <CardTitle>Technique refs</CardTitle>
        <CardDescription>
          The technique pool produced by paper ingestion. Submit a reproduce-paper proposal against
          one of these to run it.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {paper.technique_refs.length === 0 ? (
          <p className="text-sm text-[var(--color-fg-subtle)]">
            No technique refs registered (the registered state implies at least one — surface as a
            data anomaly if this is what you are seeing).
          </p>
        ) : (
          <ul className="space-y-2">
            {paper.technique_refs.map((ref) => (
              <TechniqueRefRow key={ref.technique_id} ref={ref} />
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

function TechniqueRefRow({ ref }: { ref: TechniqueRef }) {
  return (
    <li className="rounded-md border border-[var(--color-border)] bg-[var(--color-bg)] p-3">
      <div className="flex flex-wrap items-center gap-2">
        <code className="font-mono text-sm font-medium">{ref.technique_id}</code>
        {ref.name && <span className="text-sm">{ref.name}</span>}
      </div>
      {ref.description && (
        <p className="mt-1 text-sm text-[var(--color-fg-subtle)]">{ref.description}</p>
      )}
    </li>
  );
}

function PromotePartialPanel({ paper }: { paper: PaperDetail }) {
  const queryClient = useQueryClient();
  const promoteMutation = $api.useMutation("post", "/api/v1/papers/{arxiv_id}/promote-partial", {
    onSuccess: () => {
      invalidateAfterPaperPromote(queryClient, paper.arxiv_id);
      toast.success("Paper promoted to submitted");
    },
    onError: (err) => {
      toast.error(`Promote failed: ${errorMessage(err)}`);
    },
  });

  if (paper.state !== "partial") return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Promote partial</CardTitle>
        <CardDescription>
          The paper is in a partial state. Promoting fires the partial → submitted edge so the
          ingestion pipeline picks it up again.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Button
          disabled={promoteMutation.isPending}
          onClick={() => promoteMutation.mutate({ params: { path: { arxiv_id: paper.arxiv_id } } })}
        >
          {promoteMutation.isPending ? "Promoting…" : "Promote partial"}
        </Button>
      </CardContent>
    </Card>
  );
}

function ArtifactsPanel({ paper }: { paper: PaperDetail }) {
  // Paper artifacts are referenced by individual artifact_key fields rather
  // than a list endpoint. M6 wires <ArtifactFrame> against them.
  const refs: Array<[string, string | null]> = [
    ["latex_bundle", paper.latex_bundle_artifact_key],
    ["expanded_tex", paper.expanded_tex_artifact_key],
    ["extracted_text", paper.extracted_text_artifact_key],
    ["figures", paper.figures_artifact_key],
    ["technique_buffer", paper.technique_buffer_artifact_key],
    ["error_context", paper.error_context_artifact_key],
  ];
  const populated = refs.filter(([, key]) => key != null) as Array<[string, string]>;
  if (populated.length === 0) return null;
  return (
    <Card>
      <CardHeader>
        <CardTitle>Artifacts</CardTitle>
        <CardDescription>
          Paper-level artifacts produced during ingestion. Filenames render here as monospace
          identifiers; 4.M6 wires the artifact viewer.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ul className="space-y-1 text-sm">
          {populated.map(([label, key]) => (
            <li key={label} className="flex gap-2">
              <span className="min-w-[10rem] text-[var(--color-fg-subtle)]">{label}</span>
              <code className="font-mono text-xs break-all">{key}</code>
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}

function KV({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex gap-2 font-mono">
      <dt className="min-w-[10rem] text-[var(--color-fg-subtle)]">{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}
