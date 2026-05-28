import { useQueryClient } from "@tanstack/react-query";
import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { $api } from "@/lib/api/client";
import {
  errorMessage,
  extractApiError,
  invalidateAfterProposalSubmit,
  isPaperNotReady,
} from "@/lib/api/mutation-helpers";
import { proposalSubmissionSchema } from "@/lib/forms/schemas";

type SubmissionKind = "novel_technique" | "reproduce_paper";

export const Route = createFileRoute("/proposals/new")({
  component: NewProposalPage,
});

function NewProposalPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const [submissionKind, setSubmissionKind] = useState<SubmissionKind>("novel_technique");
  const [techniqueDescriptionText, setTechniqueDescriptionText] = useState("");
  const [reproduceArxivId, setReproduceArxivId] = useState("");
  const [submittedBy, setSubmittedBy] = useState("");
  const [clientErrors, setClientErrors] = useState<Record<string, string>>({});

  const mutation = $api.useMutation("post", "/api/v1/proposals", {
    onSuccess: (data) => {
      invalidateAfterProposalSubmit(queryClient);
      toast.success(`Proposal submitted (${data.id})`);
      void navigate({ to: "/proposals/$id", params: { id: data.id } });
    },
    onError: (err) => {
      if (isPaperNotReady(err)) {
        toast.error(
          "Paper is still being ingested. Wait for it to reach `registered` state, then retry.",
        );
        return;
      }
      const parsed = extractApiError(err);
      // 11-api.md §6.1: VALIDATION_ERROR may carry per-field issues. Surface
      // them alongside the toast so the user sees both.
      if (parsed.issues && parsed.issues.length > 0) {
        const fieldErrors: Record<string, string> = {};
        for (const issue of parsed.issues) {
          const last = issue.loc[issue.loc.length - 1];
          if (typeof last === "string") fieldErrors[last] = issue.msg;
        }
        setClientErrors(fieldErrors);
      }
      toast.error(parsed.message ?? "Submission failed");
    },
  });

  const candidate = useMemo(() => {
    if (submissionKind === "novel_technique") {
      return {
        submission_kind: "novel_technique" as const,
        description: techniqueDescriptionText,
      };
    }
    return {
      submission_kind: "reproduce_paper" as const,
      reproduce_paper_arxiv_id: reproduceArxivId.trim(),
    };
  }, [submissionKind, techniqueDescriptionText, reproduceArxivId]);

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setClientErrors({});

    const result = proposalSubmissionSchema.safeParse(candidate);
    if (!result.success) {
      const fieldErrors: Record<string, string> = {};
      for (const issue of result.error.issues) {
        const last = issue.path[issue.path.length - 1];
        if (typeof last === "string") fieldErrors[last] = issue.message;
      }
      setClientErrors(fieldErrors);
      return;
    }

    const body: {
      submission_kind: SubmissionKind;
      description?: string;
      reproduce_paper_arxiv_id?: string;
      submitted_by?: string | null;
    } = { submission_kind: result.data.submission_kind };

    if (result.data.submission_kind === "novel_technique") {
      body.description = result.data.description;
    } else {
      body.reproduce_paper_arxiv_id = result.data.reproduce_paper_arxiv_id;
    }
    if (submittedBy.trim().length > 0) body.submitted_by = submittedBy.trim();

    mutation.mutate({ body });
  };

  return (
    <div className="space-y-4">
      <header>
        <Link to="/proposals" className="text-xs text-[var(--color-fg-subtle)] hover:underline">
          ← All proposals
        </Link>
        <h1 className="mt-1 text-2xl font-bold tracking-tight">New proposal</h1>
        <p className="mt-1 text-sm text-[var(--color-fg-subtle)]">
          Describe the comparison you want to test. After you submit, the planner agent designs it
          into one or more comparison groups and waits for your approval before any run starts.
        </p>
      </header>

      <form onSubmit={handleSubmit} noValidate>
        <Card>
          <CardHeader>
            <CardTitle>What to compare</CardTitle>
            <CardDescription>
              Pick whether you're proposing a novel technique or reproducing a paper, then fill in
              the matching description.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-1">
              <label htmlFor="submission-kind" className="text-sm font-medium">
                Submission kind
              </label>
              <Select
                value={submissionKind}
                onValueChange={(value) => setSubmissionKind(value as SubmissionKind)}
              >
                <SelectTrigger id="submission-kind" className="w-64">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="novel_technique">Novel technique</SelectItem>
                  <SelectItem value="reproduce_paper">Reproduce paper</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {submissionKind === "novel_technique" ? (
              <div className="space-y-1">
                <label htmlFor="technique-description-text" className="text-sm font-medium">
                  Technique description
                </label>
                <Textarea
                  id="technique-description-text"
                  value={techniqueDescriptionText}
                  onChange={(e) => setTechniqueDescriptionText(e.target.value)}
                  rows={10}
                  placeholder="Describe the novel technique in free-form prose. The planner agent drafts the technique from this."
                  aria-invalid={Boolean(clientErrors.description)}
                />
                {clientErrors.description && <FieldError msg={clientErrors.description} />}
              </div>
            ) : (
              <div className="space-y-1">
                <label htmlFor="reproduce-arxiv-id" className="text-sm font-medium">
                  arXiv ID of the paper to reproduce
                </label>
                <Input
                  id="reproduce-arxiv-id"
                  value={reproduceArxivId}
                  onChange={(e) => setReproduceArxivId(e.target.value)}
                  placeholder="e.g., 2401.12345"
                  className="font-mono"
                  aria-invalid={Boolean(clientErrors.reproduce_paper_arxiv_id)}
                />
                {clientErrors.reproduce_paper_arxiv_id && (
                  <FieldError msg={clientErrors.reproduce_paper_arxiv_id} />
                )}
                <p className="text-xs text-[var(--color-fg-subtle)]">
                  The paper must already be registered. Need to ingest it first?{" "}
                  <Link to="/papers/new" className="text-[var(--color-accent)] hover:underline">
                    Ingest a new paper
                  </Link>
                  .
                </p>
              </div>
            )}

            <div className="space-y-1">
              <label htmlFor="submitted-by" className="text-sm font-medium">
                Submitted by{" "}
                <span className="text-[var(--color-fg-subtle)]">(optional, audit trail)</span>
              </label>
              <Input
                id="submitted-by"
                value={submittedBy}
                onChange={(e) => setSubmittedBy(e.target.value)}
                placeholder="your-handle"
              />
            </div>

            <div className="flex items-center gap-3 pt-2">
              <Button type="submit" disabled={mutation.isPending}>
                {mutation.isPending ? "Submitting…" : "Submit proposal"}
              </Button>
              <Link
                to="/proposals"
                className="text-sm text-[var(--color-fg-subtle)] hover:underline"
              >
                Cancel
              </Link>
            </div>

            {mutation.isError && !mutation.isPending && (
              <p className="text-sm text-[var(--color-danger)]">{errorMessage(mutation.error)}</p>
            )}
          </CardContent>
        </Card>
      </form>
    </div>
  );
}

function FieldError({ msg }: { msg: string }) {
  return <p className="text-xs text-[var(--color-danger)]">{msg}</p>;
}
