import { QueryClient } from "@tanstack/react-query";
import createFetchClient from "openapi-fetch";
import createClient from "openapi-react-query";

import type { paths } from "./generated/api-types";

// 13-frontend.md §5.3 + 12-ui-process.md §8.1: same-origin URLs everywhere.
// Vite proxies /api/* to the FastAPI dev server in dev; smai-api serves the
// SPA bundle and the API from the same origin in prod, so no env-switched
// baseUrl is needed.
export const fetchClient = createFetchClient<paths>({
  baseUrl: "/",
  // Auth header injection lands in 4.N1 when the bearer-token bootstrap is wired:
  // headers: () => window.__SMAI_TOKEN__ ? { Authorization: `Bearer ${window.__SMAI_TOKEN__}` } : {}
});

export const $api = createClient(fetchClient);

// Single QueryClient instance shared between the React tree (via
// QueryClientProvider in main.tsx) and the SSE subscriber (lib/events/sse.ts),
// which needs to call invalidateQueries / setQueryData from outside React.
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Conservative defaults; per-query overrides land where needed.
      // SSE drives fresh data via invalidation per 13-frontend.md §7;
      // the staleTime here is the fallback-when-no-SSE poll cadence.
      staleTime: 30_000,
      refetchOnWindowFocus: true,
    },
  },
});
