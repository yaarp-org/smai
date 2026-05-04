// Shared loading / error / empty-state shells. Every read page handles
// isPending and error states per the brief. The error message extracts the
// `error.error.message` from the API envelope (11-api.md §6.1) when present;
// otherwise falls back to the JS Error message or String(error).

export interface ErrorBannerProps {
  error: unknown;
  retry?: () => void;
}

interface ApiErrorEnvelope {
  error?: { message?: string; code?: string };
}

function extractMessage(error: unknown): string {
  if (error && typeof error === "object") {
    const env = error as ApiErrorEnvelope;
    if (env.error?.message) return env.error.message;
    if (error instanceof Error) return error.message;
  }
  return String(error);
}

export function ErrorBanner({ error, retry }: ErrorBannerProps) {
  return (
    <div
      role="alert"
      className="flex items-center gap-3 rounded-md border border-[var(--color-danger)]/40 bg-[color-mix(in_oklch,var(--color-danger)_8%,transparent)] px-3 py-2 text-sm text-[var(--color-danger)]"
    >
      <span className="font-medium">Error:</span>
      <span className="flex-1">{extractMessage(error)}</span>
      {retry && (
        <button
          onClick={retry}
          className="rounded border border-current px-2 py-0.5 text-xs font-medium hover:bg-[var(--color-bg-subtle)]"
        >
          Retry
        </button>
      )}
    </div>
  );
}

export function LoadingBlock({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-[var(--color-fg-subtle)]">
      <span
        className="inline-block h-3 w-3 animate-pulse rounded-full bg-[var(--color-fg-subtle)]/50"
        aria-hidden
      />
      {label}
    </div>
  );
}

export function EmptyState({ message }: { message: string }) {
  return (
    <p className="rounded-md border border-dashed border-[var(--color-border)] px-4 py-8 text-center text-sm text-[var(--color-fg-subtle)]">
      {message}
    </p>
  );
}
