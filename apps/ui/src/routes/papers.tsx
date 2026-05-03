import { createFileRoute } from "@tanstack/react-router";

import { PlaceholderPage } from "@/components/placeholder-page";

export const Route = createFileRoute("/papers")({
  component: PapersPage,
});

function PapersPage() {
  return (
    <PlaceholderPage
      title="Papers"
      description="Supporting input utility (per 13-frontend.md §11.2). List + ingest-by-arXiv CTA."
      upcomingTask="Task 4.M3"
    />
  );
}
