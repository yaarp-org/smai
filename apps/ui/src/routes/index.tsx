import { createFileRoute } from "@tanstack/react-router";

import { PlaceholderPage } from "@/components/placeholder-page";

export const Route = createFileRoute("/")({
  component: DashboardPage,
});

function DashboardPage() {
  return (
    <PlaceholderPage
      title="Dashboard"
      description="Summary of proposals, comparison groups, and recent activity (per 13-frontend.md §11.2)."
      upcomingTask="Task 4.M3"
    />
  );
}
