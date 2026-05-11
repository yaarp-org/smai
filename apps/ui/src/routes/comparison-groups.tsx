import { Outlet, createFileRoute } from "@tanstack/react-router";

// Layout route for `/comparison-groups/*`. The list view lives in
// `comparison-groups.index.tsx`; `comparison-groups.$id.tsx` (and its
// own nested entry / artifact routes) are siblings of the index.
function ComparisonGroupsLayout() {
  return <Outlet />;
}

export const Route = createFileRoute("/comparison-groups")({
  component: ComparisonGroupsLayout,
});
