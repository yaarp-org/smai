import { Outlet, createFileRoute } from "@tanstack/react-router";

// Layout route for `/proposals/*`. The list view lives in
// `proposals.index.tsx`; `proposals.new.tsx` and `proposals.$id.tsx`
// are siblings. This component is just the <Outlet/> the router renders
// children into — without it the parent list page would stay on screen
// and the child would never mount.
function ProposalsLayout() {
  return <Outlet />;
}

export const Route = createFileRoute("/proposals")({
  component: ProposalsLayout,
});
