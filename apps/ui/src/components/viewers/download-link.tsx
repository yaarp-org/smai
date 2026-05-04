import { Download } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export interface DownloadLinkProps {
  url: string;
  filename: string;
  label?: string;
  className?: string;
}

// Plain anchor with the download attribute. The smai-api artifact endpoint
// either streams bytes directly (LocalFs) or returns a 302 to a presigned URL
// (S3) per 11-api.md §5.2.4 — the browser handles both transparently.
export function DownloadLink({ url, filename, label, className }: DownloadLinkProps) {
  return (
    <div
      className={cn(
        "flex items-center justify-between gap-3 rounded-md border border-[var(--color-border)] bg-[var(--color-bg-subtle)] p-3 text-sm",
        className,
      )}
    >
      <div className="min-w-0">
        <div className="truncate font-mono text-[var(--color-fg)]">{filename}</div>
        <div className="text-xs text-[var(--color-fg-subtle)]">
          {label ?? "Binary or unsupported artifact — download to inspect."}
        </div>
      </div>
      <Button asChild variant="outline" size="sm">
        <a href={url} download={filename} rel="noopener">
          <Download aria-hidden /> Download
        </a>
      </Button>
    </div>
  );
}
