// Client-side Zod schemas for the M4 form routes. The authoritative
// validation lives server-side in smai-api-spec's Pydantic models (per
// 13-frontend.md §3.3 — "no form library day one; useState + Zod is enough");
// these schemas are UX polish so the user sees inline errors before the
// network round-trip.

import { z } from "zod";

// 11-api.md §5.2.1: SubmitProposalRequest — exactly one of the three
// description fields populated, consistent with submission_kind. We model
// the two surface forms separately and union them:
//
//   * novel_technique → free-form text. (The inline-JSON form is reserved
//     for future structured-description UX; the day-one form route only
//     ships the text path per 13-frontend.md §11.3.)
//   * reproduce_paper → arxiv_id pinned. The server returns 409 +
//     PAPER_NOT_READY when the paper isn't in `registered`.
export const proposalSubmissionSchema = z.discriminatedUnion("submission_kind", [
  z.object({
    submission_kind: z.literal("novel_technique"),
    technique_description_text: z
      .string()
      .min(1, "Technique description is required")
      .max(64_000, "Technique description must be under 64K characters"),
  }),
  z.object({
    submission_kind: z.literal("reproduce_paper"),
    reproduce_paper_arxiv_id: z.string().min(1, "arXiv ID is required"),
  }),
]);

export type ProposalSubmissionInput = z.infer<typeof proposalSubmissionSchema>;

// 11-api.md §4.2: arxiv_id format is intentionally permissive at the API
// boundary (both YYMM.NNNNN[v#] and the legacy archive/YYMM### shapes
// exist; the plugin layer enforces). The client-side regex matches both;
// it's a UX hint, not a contract gate.
export const arxivIdPattern = /^\d{4}\.\d{4,5}(v\d+)?$|^[a-z-]+\/\d{7}$/;

export const arxivIdSchema = z
  .string()
  .min(1, "arXiv ID is required")
  .max(64, "arXiv ID must be under 64 characters")
  .regex(arxivIdPattern, "Invalid arXiv ID format (e.g., 2401.12345 or cs.LG/0701234)");

export const experimentYamlSchema = z
  .string()
  .min(1, "Experiment YAML is required")
  .max(1_000_000, "Experiment YAML is unexpectedly large (>1MB)");
