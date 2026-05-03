import { createFileRoute } from "@tanstack/react-router";

import { PlaceholderPage } from "@/components/placeholder-page";

export const Route = createFileRoute("/proposals")({
  component: ProposalsPage,
});

function ProposalsPage() {
  return (
    <PlaceholderPage
      title="Proposals"
      description="Primary input verb per DEC-032. List + submit-new CTA (per 13-frontend.md §11.2)."
      upcomingTask="Task 4.M3"
    />
  );
}
