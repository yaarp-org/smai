// 13-frontend.md §11.2 helper: ISO timestamp → human-readable local time +
// short relative form ("3m ago"). All API timestamps come over the wire as
// ISO 8601 strings; this normalizes them for display.

const RELATIVE_THRESHOLDS: ReadonlyArray<
  readonly [seconds: number, divisor: number, unit: string]
> = [
  [60, 1, "s"],
  [3600, 60, "m"],
  [86_400, 3600, "h"],
  [604_800, 86_400, "d"],
] as const;

export function formatDateTime(input: string | Date | null | undefined): string {
  if (input == null) return "—";
  const d = input instanceof Date ? input : new Date(input);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function formatRelative(input: string | Date | null | undefined): string {
  if (input == null) return "—";
  const d = input instanceof Date ? input : new Date(input);
  if (Number.isNaN(d.getTime())) return "—";
  const diff = Math.floor((Date.now() - d.getTime()) / 1000);
  if (diff < 0) return "in the future";
  if (diff < 5) return "just now";
  for (const [bound, divisor, unit] of RELATIVE_THRESHOLDS) {
    if (diff < bound) return `${Math.floor(diff / divisor)}${unit} ago`;
  }
  return `${Math.floor(diff / 604_800)}w ago`;
}
