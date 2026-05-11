import { Outlet, createFileRoute } from "@tanstack/react-router";

// Layout route for `/runs/*`. List view lives in `runs.index.tsx`;
// `runs.$id.tsx` is its sibling. This is the <Outlet/> the router
// renders children into.
function RunsLayout() {
  return <Outlet />;
}

export const Route = createFileRoute("/runs")({
  component: RunsLayout,
});
