import { Link } from "@tanstack/react-router";

import { StateBadge } from "@/components/entity/state-badge";
import { TableCell, TableRow } from "@/components/ui/table";
import type { components } from "@/lib/api/generated/api-types";

type EntryWithRuns = components["schemas"]["EntryWithRuns"];

export function EntryRow({ entry }: { entry: EntryWithRuns }) {
  const runCounts = entry.runs.reduce<Record<string, number>>((acc, run) => {
    acc[run.state] = (acc[run.state] ?? 0) + 1;
    return acc;
  }, {});
  return (
    <TableRow>
      <TableCell className="font-mono text-xs">
        <Link
          to="/comparison-groups/$id/entries/$entryId"
          params={{ id: entry.cg_id, entryId: entry.id }}
          className="text-[var(--color-accent)] hover:underline"
        >
          {entry.id}
        </Link>
      </TableCell>
      <TableCell className="font-mono text-xs">{entry.technique_id ?? "—"}</TableCell>
      <TableCell>
        {entry.is_baseline ? (
          <span className="rounded bg-[var(--color-bg-subtle)] px-1.5 py-0.5 text-xs">
            baseline
          </span>
        ) : (
          <span className="text-xs text-[var(--color-fg-subtle)]">—</span>
        )}
      </TableCell>
      <TableCell>
        <StateBadge state={entry.state} />
      </TableCell>
      <TableCell className="font-mono text-xs">attempt {entry.implementation_attempt}</TableCell>
      <TableCell>
        <div className="flex flex-wrap gap-1 text-xs">
          {entry.runs.length === 0 ? (
            <span className="text-[var(--color-fg-subtle)]">no runs</span>
          ) : (
            Object.entries(runCounts).map(([state, count]) => (
              <span
                key={state}
                className="rounded bg-[var(--color-bg-subtle)] px-1.5 py-0.5 font-mono"
              >
                {state}:{count}
              </span>
            ))
          )}
        </div>
      </TableCell>
    </TableRow>
  );
}
