import { Outlet, createFileRoute } from "@tanstack/react-router";

// Layout route for `/papers/*`. List view lives in `papers.index.tsx`;
// `papers.new.tsx` and `papers.$arxivId.tsx` are siblings. This is the
// <Outlet/> the router renders children into.
function PapersLayout() {
  return <Outlet />;
}

export const Route = createFileRoute("/papers")({
  component: PapersLayout,
});
