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
