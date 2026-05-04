// Mutation invalidation matrix (per 13-frontend.md §6.4 — invalidate-on-success,
// no optimistic updates). Each helper returns the queryKey prefixes to invalidate
// after a given mutation succeeds. TanStack Query's `invalidateQueries({ queryKey })`
// matches by prefix, so passing `["get", "/api/v1/proposals"]` invalidates every
// list-page variant (any state filter, any cursor) plus any query whose key starts
// with that pair.
//
// Cross-resource matrix:
//
//   POST /api/v1/proposals (submit)              → proposals list
//   POST /api/v1/proposals/{id}/approve          → proposals list + this proposal detail + CGs list
//   POST /api/v1/proposals/{id}/reject           → proposals list + this proposal detail
//   POST /api/v1/papers (ingest)                 → papers list
//   POST /api/v1/papers/{arxiv_id}/promote-partial → papers list + this paper detail
//   POST /api/v1/experiments (compile + submit)  → CGs list (creates 1+ CGs)
//   POST /api/v1/experiments/compile             → no invalidation (pure compile, doesn't persist)
//
// Plus dashboard summary counts whenever an entity transitions, since the dashboard
// pulls from /api/v1/system/dashboard which aggregates across kinds.

import type { QueryClient, QueryKey } from "@tanstack/react-query";

const PROPOSALS_LIST: QueryKey = ["get", "/api/v1/proposals"];
const PAPERS_LIST: QueryKey = ["get", "/api/v1/papers"];
const COMPARISON_GROUPS_LIST: QueryKey = ["get", "/api/v1/comparison-groups"];
const SYSTEM_DASHBOARD: QueryKey = ["get", "/api/v1/system/dashboard"];

function proposalDetail(id: string): QueryKey {
  return ["get", "/api/v1/proposals/{proposal_id}", { params: { path: { proposal_id: id } } }];
}

function paperDetail(arxivId: string): QueryKey {
  return ["get", "/api/v1/papers/{arxiv_id}", { params: { path: { arxiv_id: arxivId } } }];
}

export function invalidateAfterProposalSubmit(qc: QueryClient): void {
  void qc.invalidateQueries({ queryKey: PROPOSALS_LIST });
  void qc.invalidateQueries({ queryKey: SYSTEM_DASHBOARD });
}

export function invalidateAfterProposalApprove(qc: QueryClient, proposalId: string): void {
  void qc.invalidateQueries({ queryKey: PROPOSALS_LIST });
  void qc.invalidateQueries({ queryKey: proposalDetail(proposalId) });
  void qc.invalidateQueries({ queryKey: COMPARISON_GROUPS_LIST });
  void qc.invalidateQueries({ queryKey: SYSTEM_DASHBOARD });
}

export function invalidateAfterProposalReject(qc: QueryClient, proposalId: string): void {
  void qc.invalidateQueries({ queryKey: PROPOSALS_LIST });
  void qc.invalidateQueries({ queryKey: proposalDetail(proposalId) });
  void qc.invalidateQueries({ queryKey: SYSTEM_DASHBOARD });
}

export function invalidateAfterPaperSubmit(qc: QueryClient): void {
  void qc.invalidateQueries({ queryKey: PAPERS_LIST });
  void qc.invalidateQueries({ queryKey: SYSTEM_DASHBOARD });
}

export function invalidateAfterPaperPromote(qc: QueryClient, arxivId: string): void {
  void qc.invalidateQueries({ queryKey: PAPERS_LIST });
  void qc.invalidateQueries({ queryKey: paperDetail(arxivId) });
  void qc.invalidateQueries({ queryKey: SYSTEM_DASHBOARD });
}

export function invalidateAfterExperimentSubmit(qc: QueryClient): void {
  void qc.invalidateQueries({ queryKey: COMPARISON_GROUPS_LIST });
  void qc.invalidateQueries({ queryKey: SYSTEM_DASHBOARD });
}

// Error-envelope extraction. openapi-fetch surfaces the parsed non-2xx body
// as `mutation.error`; smai-api emits the ErrorEnvelope shape (11-api.md §6.1)
// `{error: {code, message, issues?, retryable?}}`. We don't have a generated
// TS type for the envelope (FastAPI doesn't currently include it in the
// OpenAPI schema; surfaced as an open question), so we parse defensively here.
//
// Mirrors components/common/page-states.tsx's read-side extractMessage; kept
// separate because the mutation path also needs `code` (e.g., the
// PAPER_NOT_READY 409 special-case from 11 §13 OQ11) and `issues`.

export interface ApiErrorShape {
  code?: string;
  message?: string;
  issues?: ValidationIssue[];
  retryable?: boolean;
}

export interface ValidationIssue {
  loc: (string | number)[];
  msg: string;
  type: string;
}

export function extractApiError(error: unknown): ApiErrorShape {
  if (error && typeof error === "object" && "error" in error) {
    const inner = (error as { error?: unknown }).error;
    if (inner && typeof inner === "object") {
      return inner as ApiErrorShape;
    }
  }
  if (error instanceof Error) return { message: error.message };
  return { message: String(error) };
}

export function errorMessage(error: unknown, fallback = "Request failed"): string {
  return extractApiError(error).message ?? fallback;
}

// 11-api.md §13 OQ11 RESOLVED: 409 + code: PAPER_NOT_READY surfaced as a
// distinct user-friendly message rather than the generic envelope text.
export function isPaperNotReady(error: unknown): boolean {
  return extractApiError(error).code === "PAPER_NOT_READY";
}
