import { Outlet, createFileRoute } from "@tanstack/react-router";

// Layout route for `/comparison-groups/$id/*`. The CG detail view lives
// in `comparison-groups.$id.index.tsx`; the per-entry detail
// (`...$id.entries.$entryId.tsx`) and the artifact viewer
// (`...$id.artifacts.$.tsx`) are siblings of the index. This is the
// <Outlet/> the router renders children into.
function ComparisonGroupDetailLayout() {
  return <Outlet />;
}

export const Route = createFileRoute("/comparison-groups/$id")({
  component: ComparisonGroupDetailLayout,
});
