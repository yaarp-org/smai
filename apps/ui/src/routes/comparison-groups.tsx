import { createFileRoute } from "@tanstack/react-router";

import { PlaceholderPage } from "@/components/placeholder-page";

export const Route = createFileRoute("/comparison-groups")({
  component: ComparisonGroupsPage,
});

function ComparisonGroupsPage() {
  return (
    <PlaceholderPage
      title="Comparison Groups"
      description="CG list with state filter (per 13-frontend.md §11.2)."
      upcomingTask="Task 4.M3"
    />
  );
}
