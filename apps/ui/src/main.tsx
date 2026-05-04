import { QueryClientProvider } from "@tanstack/react-query";
import { RouterProvider, createRouter } from "@tanstack/react-router";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "@/styles/globals.css";

import { queryClient } from "@/lib/api/client";
import { startSseSubscription, stopSseSubscription } from "@/lib/events/sse";

import { routeTree } from "./routeTree.gen";

const router = createRouter({
  routeTree,
  defaultPreload: "intent",
  context: { queryClient },
});

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}

const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error("Root element #root not found in index.html");
}

// Open one EventSource against /api/v1/events for the lifetime of the page.
// The subscriber is idempotent and lives outside React; cleanup on
// `beforeunload` is best-effort tidiness — the browser will close the socket
// regardless when the tab unloads.
startSseSubscription();
window.addEventListener("beforeunload", stopSseSubscription);

createRoot(rootElement).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>,
);
