import { useQueryClient } from "@tanstack/react-query";
import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { JsonTree } from "@/components/viewers";
import { $api } from "@/lib/api/client";
import {
  errorMessage,
  extractApiError,
  invalidateAfterExperimentSubmit,
} from "@/lib/api/mutation-helpers";
import { experimentYamlSchema } from "@/lib/forms/schemas";
import type { components } from "@/lib/api/generated/api-types";

type CompileResponse = components["schemas"]["CompileExperimentResponse"];

export const Route = createFileRoute("/experiments/new")({
  component: NewExperimentPage,
});

function NewExperimentPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const [yamlText, setYamlText] = useState("");
  const [clientError, setClientError] = useState<string | null>(null);
  const [preview, setPreview] = useState<CompileResponse | null>(null);

  const compileMutation = $api.useMutation("post", "/api/v1/experiments/compile", {
    onSuccess: (data) => {
      setPreview(data);
      toast.success(`Compiled ${data.compilations.length} CG(s) — preview only, not persisted`);
    },
    onError: (err) => {
      setPreview(null);
      const parsed = extractApiError(err);
      if (parsed.issues && parsed.issues.length > 0) {
        const summary = parsed.issues
          .map((issue) => `${issue.loc.join(".")}: ${issue.msg}`)
          .join("; ");
        toast.error(`Validation: ${summary}`);
        return;
      }
      toast.error(parsed.message ?? "Compile failed");
    },
  });

  const submitMutation = $api.useMutation("post", "/api/v1/experiments", {
    onSuccess: (data) => {
      invalidateAfterExperimentSubmit(queryClient);
      const created = data.cgs.length;
      toast.success(`Experiment submitted (${created} CG${created === 1 ? "" : "s"} created)`);
      const first = data.cgs[0];
      if (first) {
        void navigate({ to: "/comparison-groups/$id", params: { id: first.cg_id } });
      } else {
        void navigate({ to: "/comparison-groups" });
      }
    },
    onError: (err) => {
      const parsed = extractApiError(err);
      if (parsed.issues && parsed.issues.length > 0) {
        const summary = parsed.issues
          .map((issue) => `${issue.loc.join(".")}: ${issue.msg}`)
          .join("; ");
        toast.error(`Validation: ${summary}`);
        return;
      }
      toast.error(parsed.message ?? "Submit failed");
    },
  });

  const validate = (): string | null => {
    const result = experimentYamlSchema.safeParse(yamlText);
    if (!result.success) {
      const first = result.error.issues[0];
      return first?.message ?? "Invalid input";
    }
    return null;
  };

  const handleCompile = () => {
    setClientError(null);
    const errMsg = validate();
    if (errMsg) {
      setClientError(errMsg);
      return;
    }
    compileMutation.mutate({ body: { definition_text: yamlText } });
  };

  const handleSubmit = () => {
    setClientError(null);
    const errMsg = validate();
    if (errMsg) {
      setClientError(errMsg);
      return;
    }
    submitMutation.mutate({ body: { definition_text: yamlText } });
  };

  const busy = compileMutation.isPending || submitMutation.isPending;

  return (
    <div className="space-y-4">
      <header>
        <Link
          to="/comparison-groups"
          className="text-xs text-[var(--color-fg-subtle)] hover:underline"
        >
          ← Comparison groups
        </Link>
        <h1 className="mt-1 text-2xl font-bold tracking-tight">New experiment</h1>
        <p className="mt-1 text-sm text-[var(--color-fg-subtle)]">
          The HTTP analog of <code className="font-mono">smai run</code>. Compile previews the four
          contract artifacts (methodology-only, no persistence). Compile + submit additionally
          persists the artifacts and creates 1+ CG records in{" "}
          <code className="font-mono">draft</code>.
        </p>
      </header>

      <Card>
        <CardHeader>
          <CardTitle>Experiment YAML</CardTitle>
          <CardDescription>
            Paste a complete experiment definition. The methodology compiler produces the four
            contract artifacts (<code className="font-mono">ExperimentPlan</code>,{" "}
            <code className="font-mono">HarnessContract</code>,{" "}
            <code className="font-mono">TechniqueContract[]</code>,{" "}
            <code className="font-mono">ValidationConfig</code>).
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-1">
            <label htmlFor="experiment-yaml" className="text-sm font-medium">
              YAML
            </label>
            <Textarea
              id="experiment-yaml"
              value={yamlText}
              onChange={(e) => setYamlText(e.target.value)}
              rows={20}
              className="font-mono text-xs"
              placeholder={`experiment:\n  name: my-experiment\n  ...`}
              aria-invalid={Boolean(clientError)}
            />
            {clientError && <p className="text-xs text-[var(--color-danger)]">{clientError}</p>}
          </div>

          <div className="flex items-center gap-3 pt-2">
            <Button type="button" variant="outline" onClick={handleCompile} disabled={busy}>
              {compileMutation.isPending ? "Compiling…" : "Compile (preview)"}
            </Button>
            <Button type="button" onClick={handleSubmit} disabled={busy}>
              {submitMutation.isPending ? "Submitting…" : "Compile + submit"}
            </Button>
            <Link
              to="/comparison-groups"
              className="text-sm text-[var(--color-fg-subtle)] hover:underline"
            >
              Cancel
            </Link>
          </div>

          {compileMutation.isError && !compileMutation.isPending && (
            <p className="text-sm text-[var(--color-danger)]">
              Compile error: {errorMessage(compileMutation.error)}
            </p>
          )}
          {submitMutation.isError && !submitMutation.isPending && (
            <p className="text-sm text-[var(--color-danger)]">
              Submit error: {errorMessage(submitMutation.error)}
            </p>
          )}
        </CardContent>
      </Card>

      {preview && <CompilePreview compilations={preview.compilations} />}
    </div>
  );
}

function CompilePreview({
  compilations,
}: {
  compilations: components["schemas"]["CompiledArtifacts"][];
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Compile preview</CardTitle>
        <CardDescription>
          {compilations.length} comparison-group{compilations.length === 1 ? "" : "s"} compiled.
          Nothing has been persisted — click &ldquo;Compile + submit&rdquo; to register.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {compilations.map((c, i) => (
          <div
            key={c.cg_id || `cg-${i}`}
            className="rounded-md border border-[var(--color-border)] bg-[var(--color-bg)] p-3"
          >
            <div className="mb-2 text-xs font-medium text-[var(--color-fg-subtle)]">
              {c.cg_id ? `cg_id: ${c.cg_id}` : `compilation ${i + 1}`}
            </div>
            <JsonTree
              data={{
                experiment_plan: c.experiment_plan,
                harness_contract: c.harness_contract,
                technique_contracts: c.technique_contracts,
                validation_config: c.validation_config,
              }}
              defaultExpandDepth={1}
            />
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
