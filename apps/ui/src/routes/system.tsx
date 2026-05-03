import { createFileRoute } from "@tanstack/react-router";

import { PlaceholderPage } from "@/components/placeholder-page";

export const Route = createFileRoute("/system")({
  component: SystemPage,
});

function SystemPage() {
  return (
    <PlaceholderPage
      title="System"
      description="Plugin status, version, verify, migrate-status (per 13-frontend.md §11.2)."
      upcomingTask="Task 4.M3"
    />
  );
}
