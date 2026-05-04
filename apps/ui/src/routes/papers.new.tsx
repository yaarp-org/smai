import { useQueryClient } from "@tanstack/react-query";
import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { $api } from "@/lib/api/client";
import {
  errorMessage,
  extractApiError,
  invalidateAfterPaperSubmit,
} from "@/lib/api/mutation-helpers";
import { arxivIdSchema } from "@/lib/forms/schemas";

export const Route = createFileRoute("/papers/new")({
  component: NewPaperPage,
});

function NewPaperPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const [arxivId, setArxivId] = useState("");
  const [clientError, setClientError] = useState<string | null>(null);

  const mutation = $api.useMutation("post", "/api/v1/papers", {
    onSuccess: (data) => {
      invalidateAfterPaperSubmit(queryClient);
      toast.success(`Paper submitted (arXiv: ${data.arxiv_id})`);
      void navigate({ to: "/papers/$arxivId", params: { arxivId: data.arxiv_id } });
    },
    onError: (err) => {
      const parsed = extractApiError(err);
      // VALIDATION_ERROR: surface first issue inline.
      if (parsed.issues && parsed.issues.length > 0) {
        const first = parsed.issues[0];
        if (first) setClientError(first.msg);
      }
      toast.error(parsed.message ?? "Submission failed");
    },
  });

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setClientError(null);

    const trimmed = arxivId.trim();
    const result = arxivIdSchema.safeParse(trimmed);
    if (!result.success) {
      const first = result.error.issues[0];
      if (first) setClientError(first.message);
      return;
    }

    mutation.mutate({ body: { arxiv_id: result.data } });
  };

  return (
    <div className="space-y-4">
      <header>
        <Link to="/papers" className="text-xs text-[var(--color-fg-subtle)] hover:underline">
          ← All papers
        </Link>
        <h1 className="mt-1 text-2xl font-bold tracking-tight">Ingest a paper</h1>
        <p className="mt-1 text-sm text-[var(--color-fg-subtle)]">
          Supporting input utility per DEC-032 — fetch + parse + screen + plan, then register the
          paper&apos;s technique pool. Idempotent: resubmitting the same arXiv ID promotes a{" "}
          <code className="font-mono">partial</code> record or no-ops on an in-flight one.
        </p>
      </header>

      <form onSubmit={handleSubmit} noValidate>
        <Card>
          <CardHeader>
            <CardTitle>arXiv ID</CardTitle>
            <CardDescription>
              Paste the modern (e.g., <code className="font-mono">2401.12345</code>) or legacy
              (e.g., <code className="font-mono">cs.LG/0701234</code>) form.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-1">
              <label htmlFor="arxiv-id" className="text-sm font-medium">
                arXiv ID
              </label>
              <Input
                id="arxiv-id"
                value={arxivId}
                onChange={(e) => setArxivId(e.target.value)}
                placeholder="2401.12345"
                className="font-mono"
                autoFocus
                aria-invalid={Boolean(clientError)}
              />
              {clientError && <p className="text-xs text-[var(--color-danger)]">{clientError}</p>}
            </div>

            <div className="flex items-center gap-3 pt-2">
              <Button type="submit" disabled={mutation.isPending}>
                {mutation.isPending ? "Ingesting…" : "Ingest paper"}
              </Button>
              <Link to="/papers" className="text-sm text-[var(--color-fg-subtle)] hover:underline">
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
