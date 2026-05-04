import { useEffect, useState } from "react";

import { cn } from "@/lib/utils";

import { CodeViewer } from "./code-viewer";
import { DownloadLink } from "./download-link";
import { JsonTree } from "./json-tree";
import { MarkdownViewer } from "./markdown-viewer";
import { detectLanguage } from "./shiki-singleton";

export interface ArtifactFrameProps {
  cgId: string;
  path: string;
  className?: string;
}

interface FetchState<T> {
  data?: T;
  error?: string;
  loading: boolean;
}

function buildArtifactUrl(cgId: string, path: string): string {
  // Re-encode each path segment but preserve "/" so smai-api's
  // {path:path} matcher receives the original layout.
  const encoded = path.split("/").map(encodeURIComponent).join("/");
  return `/api/v1/comparison-groups/${encodeURIComponent(cgId)}/artifacts/${encoded}`;
}

function basename(path: string): string {
  const slash = path.lastIndexOf("/");
  return slash >= 0 ? path.slice(slash + 1) : path;
}

function useArtifactText(url: string): FetchState<string> {
  const [state, setState] = useState<FetchState<string>>({ loading: true });
  useEffect(() => {
    let cancelled = false;
    setState({ loading: true });
    fetch(url, { credentials: "same-origin" })
      .then(async (r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.text();
      })
      .then((text) => {
        if (!cancelled) setState({ loading: false, data: text });
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setState({
            loading: false,
            error: err instanceof Error ? err.message : "Fetch failed",
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [url]);
  return state;
}

function FrameSkeleton({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "h-32 animate-pulse rounded-md border border-[var(--color-border)] bg-[var(--color-bg-subtle)]",
        className,
      )}
      aria-busy="true"
    />
  );
}

function FrameError({ message, className }: { message: string; className?: string }) {
  return (
    <div
      role="alert"
      className={cn(
        "rounded-md border border-[var(--color-danger)] bg-[var(--color-bg-subtle)] p-3 text-sm text-[var(--color-danger)]",
        className,
      )}
    >
      Failed to load artifact: {message}
    </div>
  );
}

function JsonViewerLoader({ url, className }: { url: string; className?: string }) {
  const { data, error, loading } = useArtifactText(url);
  if (loading) return <FrameSkeleton className={className} />;
  if (error) return <FrameError message={error} className={className} />;
  if (data === undefined) return <FrameError message="empty response" className={className} />;
  let parsed: unknown;
  try {
    parsed = JSON.parse(data);
  } catch (e) {
    const message = e instanceof Error ? e.message : "invalid JSON";
    // Fallback to raw text if JSON parse fails — the artifact may have been
    // truncated / corrupted; show what we got rather than swallowing it.
    return (
      <div className={cn("space-y-2", className)}>
        <FrameError message={`JSON parse error: ${message}`} />
        <CodeViewer code={data} language="json" />
      </div>
    );
  }
  return <JsonTree data={parsed} className={className} />;
}

function MarkdownViewerLoader({ url, className }: { url: string; className?: string }) {
  const { data, error, loading } = useArtifactText(url);
  if (loading) return <FrameSkeleton className={className} />;
  if (error) return <FrameError message={error} className={className} />;
  return <MarkdownViewer content={data ?? ""} className={className} />;
}

function CodeViewerLoader({
  url,
  language,
  className,
}: {
  url: string;
  language: string | null;
  className?: string;
}) {
  const { data, error, loading } = useArtifactText(url);
  if (loading) return <FrameSkeleton className={className} />;
  if (error) return <FrameError message={error} className={className} />;
  return <CodeViewer code={data ?? ""} language={language} className={className} />;
}

const CODE_EXTENSIONS = /\.(py|tex|sh|bash|yaml|yml|toml|js|ts|tsx|jsx)$/i;

export function ArtifactFrame({ cgId, path, className }: ArtifactFrameProps) {
  const url = buildArtifactUrl(cgId, path);
  const lower = path.toLowerCase();

  if (lower.endsWith(".json")) {
    return <JsonViewerLoader url={url} className={className} />;
  }
  if (lower.endsWith(".md") || lower.endsWith(".markdown")) {
    return <MarkdownViewerLoader url={url} className={className} />;
  }
  if (CODE_EXTENSIONS.test(lower)) {
    return <CodeViewerLoader url={url} language={detectLanguage(path)} className={className} />;
  }
  return <DownloadLink url={url} filename={basename(path)} className={className} />;
}
