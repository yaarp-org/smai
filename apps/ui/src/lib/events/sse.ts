// EventSource subscriber that bridges /api/v1/events to TanStack Query
// per `13-frontend.md` §7 (SSE-as-invalidator) and `11-api.md` §8.
//
// Wire format (`11` §8.1):
//   id: <int>
//   event: state_change | worker_heartbeat | refetch_all
//   data: <JSON>
//
// state_change → invalidate the matching detail + list query keys (§7.1).
// worker_heartbeat → push directly via setQueryData (small, frequent, no
// REST endpoint to refetch — the documented exception per §7.2).
// refetch_all → ring-buffer overflow sentinel; nuke the whole cache.

import { queryClient } from "@/lib/api/client";
import type { components } from "@/lib/api/generated/api-types";

// EntityKind is reused across the API; reuse the RecentActivityItem.kind enum
// so the discriminator stays in sync with the OpenAPI spec.
type EntityKind = components["schemas"]["RecentActivityItem"]["kind"];

// SSE event payloads — `StateChangeEvent` / `WorkerHeartbeatEvent` shapes per
// `smai_api_spec.events`. Not exposed in the OpenAPI components surface (SSE
// payloads aren't OpenAPI bodies), so they're declared inline here.
export type StateChangeEvent = {
  kind: EntityKind;
  id: string;
  from: string;
  to: string;
  ts: string;
};

export type WorkerHeartbeatEvent = {
  cycle_id: number;
  cycles_processed: number;
  ts: string;
};

export type SseConnectionStatus = "connecting" | "open" | "closed";

export const WORKER_HEARTBEAT_QUERY_KEY = ["worker_heartbeat"] as const;
export const SSE_STATUS_QUERY_KEY = ["sse_status"] as const;

let eventSource: EventSource | null = null;

export function startSseSubscription(): void {
  if (eventSource) return; // idempotent

  // Bearer-mode deferral — Approach A per the 4.M5 brief. Native EventSource
  // can't carry custom headers; the polyfill swap (@microsoft/fetch-event-source)
  // is post-M5 backlog. Default deployments have auth.enabled=false, so this
  // branch is dead today. Surfaces a clear log line if it ever fires.
  if (typeof window !== "undefined" && window.__SMAI_TOKEN__) {
    console.warn(
      "[smai/sse] bearer-token mode detected; live updates disabled. " +
        "Polyfill swap pending (see 13-frontend.md §7.3).",
    );
    setStatus("closed");
    return;
  }

  setStatus("connecting");
  eventSource = new EventSource("/api/v1/events");

  eventSource.addEventListener("open", () => {
    setStatus("open");
  });

  eventSource.addEventListener("state_change", (e) => {
    try {
      const event = JSON.parse((e as MessageEvent).data) as StateChangeEvent;
      invalidateForStateChange(event);
    } catch (err) {
      console.warn("[smai/sse] malformed state_change payload", err);
    }
  });

  eventSource.addEventListener("worker_heartbeat", (e) => {
    try {
      const event = JSON.parse((e as MessageEvent).data) as WorkerHeartbeatEvent;
      queryClient.setQueryData(WORKER_HEARTBEAT_QUERY_KEY, event);
    } catch (err) {
      console.warn("[smai/sse] malformed worker_heartbeat payload", err);
    }
  });

  // Sentinel: server emits `event: refetch_all` (with no body) when its ring
  // buffer overflowed and we missed events. The SPA's recourse is to drop
  // every cached query; TanStack Query refetches on next render.
  eventSource.addEventListener("refetch_all", () => {
    queryClient.invalidateQueries();
  });

  // Browser-native EventSource auto-reconnects on transient errors using the
  // last seen `id:` line as a `Last-Event-ID:` header (`11` §8.3 — server
  // replays from its in-memory ring buffer). We only need to surface the gap
  // in the connection-status indicator; no manual retry loop.
  eventSource.addEventListener("error", () => {
    setStatus(eventSource?.readyState === EventSource.CLOSED ? "closed" : "connecting");
  });
}

export function stopSseSubscription(): void {
  eventSource?.close();
  eventSource = null;
  setStatus("closed");
}

function setStatus(status: SseConnectionStatus): void {
  queryClient.setQueryData(SSE_STATUS_QUERY_KEY, status);
}

function invalidateForStateChange(event: StateChangeEvent): void {
  // Per `13-frontend.md` §7.1: invalidate the matching detail key AND the
  // list key. TanStack Query does prefix-match by default, so passing the
  // path-template-only key (e.g. `["get", "/api/v1/proposals/{proposal_id}"]`)
  // matches every detail-query init under that path without us needing to
  // know the specific {proposal_id}/{cg_id}/etc the user is currently
  // viewing. Cross-resource invalidations are intentional — state changes on
  // a CG dirty the parent proposal's child-CG list and the dashboard counts;
  // run state changes dirty the embedding CG detail; etc.
  switch (event.kind) {
    case "proposal":
      queryClient.invalidateQueries({ queryKey: ["get", "/api/v1/proposals/{proposal_id}"] });
      queryClient.invalidateQueries({ queryKey: ["get", "/api/v1/proposals"] });
      queryClient.invalidateQueries({ queryKey: ["get", "/api/v1/system/dashboard"] });
      break;
    case "paper":
      queryClient.invalidateQueries({ queryKey: ["get", "/api/v1/papers/{arxiv_id}"] });
      queryClient.invalidateQueries({ queryKey: ["get", "/api/v1/papers"] });
      queryClient.invalidateQueries({ queryKey: ["get", "/api/v1/system/dashboard"] });
      break;
    case "comparison_group":
      queryClient.invalidateQueries({ queryKey: ["get", "/api/v1/comparison-groups/{cg_id}"] });
      queryClient.invalidateQueries({ queryKey: ["get", "/api/v1/comparison-groups"] });
      // CG state changes also dirty the parent proposal detail (which lists
      // child CGs) and the dashboard summary counts.
      queryClient.invalidateQueries({ queryKey: ["get", "/api/v1/proposals/{proposal_id}"] });
      queryClient.invalidateQueries({ queryKey: ["get", "/api/v1/proposals"] });
      queryClient.invalidateQueries({ queryKey: ["get", "/api/v1/system/dashboard"] });
      break;
    case "entry":
      // Entry payloads carry the entry id, not the parent cg_id, so we
      // invalidate the CG detail prefix-match (covers every CG the user has
      // open). The entry-detail nested route is also covered by this prefix.
      queryClient.invalidateQueries({ queryKey: ["get", "/api/v1/comparison-groups/{cg_id}"] });
      break;
    case "run":
      queryClient.invalidateQueries({ queryKey: ["get", "/api/v1/runs/{run_id}"] });
      queryClient.invalidateQueries({ queryKey: ["get", "/api/v1/runs"] });
      // Runs are also embedded in CG detail (`11` §5.2.2); invalidate it.
      queryClient.invalidateQueries({ queryKey: ["get", "/api/v1/comparison-groups/{cg_id}"] });
      queryClient.invalidateQueries({ queryKey: ["get", "/api/v1/system/dashboard"] });
      break;
  }
}

declare global {
  interface Window {
    __SMAI_TOKEN__?: string;
  }
}
