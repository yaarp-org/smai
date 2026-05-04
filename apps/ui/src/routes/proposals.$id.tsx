import { useQueryClient } from "@tanstack/react-query";
import { createFileRoute, Link } from "@tanstack/react-router";
import { toast } from "sonner";

import { ErrorBanner, LoadingBlock } from "@/components/common/page-states";
import { StateBadge } from "@/components/entity/state-badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { $api } from "@/lib/api/client";
import {
  errorMessage,
  invalidateAfterProposalApprove,
  invalidateAfterProposalReject,
} from "@/lib/api/mutation-helpers";
import { formatDateTime, formatRelative } from "@/lib/format/datetime";
import type { components } from "@/lib/api/generated/api-types";

type ProposalDetail = components["schemas"]["ProposalDetailResponse"];

export const Route = createFileRoute("/proposals/$id")({
  component: ProposalDetailPage,
});

function ProposalDetailPage() {
  const { id } = Route.useParams();
  const query = $api.useQuery("get", "/api/v1/proposals/{proposal_id}", {
    params: { path: { proposal_id: id } },
  });

  return (
    <div className="space-y-4">
      <header>
        <Link to="/proposals" className="text-xs text-[var(--color-fg-subtle)] hover:underline">
          ← All proposals
        </Link>
        <h1 className="mt-1 text-2xl font-bold tracking-tight">
          Proposal <code className="font-mono text-base">{id}</code>
        </h1>
      </header>

      {query.isPending && <LoadingBlock />}
      {query.error && <ErrorBanner error={query.error} />}

      {query.data && <ProposalDetailContent proposal={query.data} />}
    </div>
  );
}

function ProposalDetailContent({ proposal }: { proposal: ProposalDetail }) {
  return (
    <div className="space-y-4">
      <ProposalHeader proposal={proposal} />
      <ProposalContextPanels proposal={proposal} />
      <DesignPlanPanel proposal={proposal} />
      <HumanGatePanel proposal={proposal} />
      <ChildCGsList proposal={proposal} />
    </div>
  );
}

function ProposalHeader({ proposal }: { proposal: ProposalDetail }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-center gap-2">
          <CardTitle>Header</CardTitle>
          <StateBadge state={proposal.state} />
          <span className="rounded bg-[var(--color-bg-subtle)] px-1.5 py-0.5 font-mono text-xs text-[var(--color-fg-subtle)]">
            {proposal.submission_kind}
          </span>
        </div>
        <CardDescription>Pipeline-tracking entity audit + decision state.</CardDescription>
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-1 gap-2 text-sm sm:grid-cols-2">
          <KV label="created_at" value={formatDateTime(proposal.created_at)} />
          <KV
            label="updated_at"
            value={`${formatDateTime(proposal.updated_at)} (${formatRelative(proposal.updated_at)})`}
          />
          <KV label="submitted_by" value={proposal.submitted_by ?? "—"} />
          <KV label="design_attempt" value={String(proposal.design_attempt)} />
          <KV label="registration_attempt" value={String(proposal.registration_attempt)} />
          <KV label="user_decision" value={proposal.user_decision ?? "—"} />
          <KV
            label="user_decided_at"
            value={proposal.user_decided_at ? formatDateTime(proposal.user_decided_at) : "—"}
          />
          {proposal.last_error && (
            <div className="sm:col-span-2">
              <span className="text-[var(--color-fg-subtle)]">last_error</span>
              <pre className="mt-1 overflow-x-auto rounded bg-[color-mix(in_oklch,var(--color-danger)_10%,transparent)] p-2 font-mono text-xs text-[var(--color-danger)]">
                {proposal.last_error}
              </pre>
            </div>
          )}
        </dl>
      </CardContent>
    </Card>
  );
}

function ProposalContextPanels({ proposal }: { proposal: ProposalDetail }) {
  if (proposal.submission_kind === "reproduce_paper") {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Paper reference</CardTitle>
          <CardDescription>
            This proposal is a reproduce-paper submission per DEC-032.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {proposal.reproduce_paper_arxiv_id ? (
            <Link
              to="/papers/$arxivId"
              params={{ arxivId: proposal.reproduce_paper_arxiv_id }}
              className="font-mono text-sm text-[var(--color-accent)] hover:underline"
            >
              arXiv: {proposal.reproduce_paper_arxiv_id} →
            </Link>
          ) : (
            <p className="text-sm text-[var(--color-fg-subtle)]">No arXiv ID set.</p>
          )}
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Technique description</CardTitle>
        <CardDescription>
          The submitted technique description body (persisted as an artifact).
        </CardDescription>
      </CardHeader>
      <CardContent>
        {proposal.technique_description_artifact_key ? (
          // 4.M6 swaps this monospace key for an inline ArtifactFrame
          // rendering the JSON or text body.
          <p className="text-sm">
            artifact:{" "}
            <code className="font-mono text-xs">{proposal.technique_description_artifact_key}</code>
          </p>
        ) : (
          <p className="text-sm text-[var(--color-fg-subtle)]">No artifact key recorded yet.</p>
        )}
      </CardContent>
    </Card>
  );
}

function DesignPlanPanel({ proposal }: { proposal: ProposalDetail }) {
  // The design plan is produced once the planner reaches `designed`. Until
  // then the artifact key is null. M6 ships <ArtifactFrame> which renders
  // the JSON; for M3 we just show the artifact reference.
  if (!proposal.design_plan_artifact_key) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Design plan</CardTitle>
          <CardDescription>
            Produced by the planner agent on the proposal_submitted → designing → designed path.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-[var(--color-fg-subtle)]">
            Design plan not yet produced (state: <code className="font-mono">{proposal.state}</code>
            ).
          </p>
        </CardContent>
      </Card>
    );
  }
  return (
    <Card>
      <CardHeader>
        <CardTitle>Design plan</CardTitle>
        <CardDescription>The planner's design plan for this proposal.</CardDescription>
      </CardHeader>
      <CardContent>
        <p className="text-sm">
          artifact: <code className="font-mono text-xs">{proposal.design_plan_artifact_key}</code>
        </p>
        <p className="mt-2 text-xs text-[var(--color-fg-subtle)]">
          4.M6 wires the inline JSON viewer; the proposal-artifacts URL space is itself an open
          question (proposals do not currently expose a /proposals/&lt;id&gt;/artifacts/ namespace
          in the API contract).
        </p>
      </CardContent>
    </Card>
  );
}

function HumanGatePanel({ proposal }: { proposal: ProposalDetail }) {
  // The gate becomes visible at `designed` with no decision yet recorded.
  // Approving fires the synchronous designed → registered transition (1+ CG
  // inserts in one transaction, per 08 §3.3); rejecting fires designed →
  // rejected. Both paths invalidate the proposal detail + list; approval
  // additionally invalidates the CGs list since CGs are created.
  const queryClient = useQueryClient();

  const approveMutation = $api.useMutation("post", "/api/v1/proposals/{proposal_id}/approve", {
    onSuccess: (data) => {
      invalidateAfterProposalApprove(queryClient, proposal.id);
      const created = data.cg_ids.length;
      toast.success(
        `Proposal approved (${created} comparison-group${created === 1 ? "" : "s"} created)`,
      );
    },
    onError: (err) => {
      toast.error(`Approval failed: ${errorMessage(err)}`);
    },
  });

  const rejectMutation = $api.useMutation("post", "/api/v1/proposals/{proposal_id}/reject", {
    onSuccess: () => {
      invalidateAfterProposalReject(queryClient, proposal.id);
      toast.success("Proposal rejected");
    },
    onError: (err) => {
      toast.error(`Reject failed: ${errorMessage(err)}`);
    },
  });

  if (proposal.state !== "designed") return null;
  if (proposal.user_decision != null) return null;

  const busy = approveMutation.isPending || rejectMutation.isPending;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Human gate</CardTitle>
        <CardDescription>
          The proposal is awaiting your decision. Approving atomically registers 1+ comparison
          groups; rejecting drops the proposal.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex gap-2">
          <Button
            disabled={busy}
            onClick={() =>
              approveMutation.mutate({
                params: { path: { proposal_id: proposal.id } },
              })
            }
          >
            {approveMutation.isPending ? "Approving…" : "Approve"}
          </Button>
          <Button
            variant="outline"
            disabled={busy}
            onClick={() =>
              rejectMutation.mutate({
                params: { path: { proposal_id: proposal.id } },
                body: null,
              })
            }
          >
            {rejectMutation.isPending ? "Rejecting…" : "Reject"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

function ChildCGsList({ proposal }: { proposal: ProposalDetail }) {
  if (proposal.registered_cg_ids.length === 0) return null;
  return (
    <Card>
      <CardHeader>
        <CardTitle>Registered comparison groups</CardTitle>
        <CardDescription>
          The CGs this proposal produced on the designed → registered transition.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ul className="space-y-1">
          {proposal.registered_cg_ids.map((cgId) => (
            <li key={cgId}>
              <Link
                to="/comparison-groups/$id"
                params={{ id: cgId }}
                className="font-mono text-sm text-[var(--color-accent)] hover:underline"
              >
                {cgId} →
              </Link>
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
