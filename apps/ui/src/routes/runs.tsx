import { createFileRoute } from "@tanstack/react-router";

import { PlaceholderPage } from "@/components/placeholder-page";

export const Route = createFileRoute("/runs")({
  component: RunsPage,
});

function RunsPage() {
  return (
    <PlaceholderPage
      title="Runs"
      description="Cross-CG run list with state + cg/entry filters (per 13-frontend.md §11.2)."
      upcomingTask="Task 4.M3"
    />
  );
}
