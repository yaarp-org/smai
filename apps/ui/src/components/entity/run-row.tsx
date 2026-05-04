import { Link } from "@tanstack/react-router";

import { StateBadge } from "@/components/entity/state-badge";
import { TableCell, TableRow } from "@/components/ui/table";
import { formatRelative } from "@/lib/format/datetime";
import { formatDuration } from "@/lib/format/duration";
import type { components } from "@/lib/api/generated/api-types";

type RunSummary = components["schemas"]["RunSummary"];

export function RunRow({ run }: { run: RunSummary }) {
  return (
    <TableRow>
      <TableCell className="font-mono text-xs">
        <Link
          to="/runs/$id"
          params={{ id: run.id }}
          className="text-[var(--color-accent)] hover:underline"
        >
          {run.id}
        </Link>
      </TableCell>
      <TableCell>
        <StateBadge state={run.state} />
      </TableCell>
      <TableCell className="font-mono">{run.seed}</TableCell>
      <TableCell className="font-mono">{formatDuration(run.duration_seconds)}</TableCell>
      <TableCell className="font-mono">attempt {run.run_attempt}</TableCell>
      <TableCell className="text-xs text-[var(--color-fg-subtle)]">
        {formatRelative(run.updated_at)}
      </TableCell>
    </TableRow>
  );
}
