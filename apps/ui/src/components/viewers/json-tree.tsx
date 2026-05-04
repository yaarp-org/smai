import { Link } from "@tanstack/react-router";
import { useState } from "react";

import { cn } from "@/lib/utils";

export type LinkResolver = (key: string, value: unknown) => string | null;

export interface JsonTreeProps {
  data: unknown;
  rootLabel?: string;
  linkResolver?: LinkResolver;
  defaultExpandDepth?: number;
  className?: string;
}

// Identifier-shaped SMAI keys → matching detail-page route. Reusable consumers
// can override or extend this map by passing their own linkResolver.
const DEFAULT_LINK_PATTERNS: Record<string, (id: string) => string> = {
  cg_id: (id) => `/comparison-groups/${id}`,
  comparison_group_id: (id) => `/comparison-groups/${id}`,
  proposal_id: (id) => `/proposals/${id}`,
  arxiv_id: (id) => `/papers/${id}`,
  entry_id: (id) => `/runs/${id}`,
  run_id: (id) => `/runs/${id}`,
};

export const defaultLinkResolver: LinkResolver = (key, value) => {
  if (typeof value !== "string" || value.length === 0) return null;
  const pattern = DEFAULT_LINK_PATTERNS[key];
  return pattern ? pattern(value) : null;
};

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function summarize(value: unknown): string {
  if (Array.isArray(value)) return `Array(${value.length})`;
  if (isPlainObject(value)) return `{${Object.keys(value).length}}`;
  return "";
}

interface NodeProps {
  label: string;
  value: unknown;
  depth: number;
  defaultExpandDepth: number;
  linkResolver: LinkResolver;
}

function PrimitiveValue({
  label,
  value,
  linkResolver,
}: {
  label: string;
  value: unknown;
  linkResolver: LinkResolver;
}) {
  const href = linkResolver(label, value);
  if (typeof value === "string") {
    if (href) {
      return (
        <Link to={href} className="text-[var(--color-accent)] underline-offset-2 hover:underline">
          &quot;{value}&quot;
        </Link>
      );
    }
    return <span className="text-[var(--color-success)]">&quot;{value}&quot;</span>;
  }
  if (typeof value === "number") {
    return <span className="text-[var(--color-accent)]">{value}</span>;
  }
  if (typeof value === "boolean") {
    return <span className="text-[var(--color-warning)]">{String(value)}</span>;
  }
  if (value === null) {
    return <span className="text-[var(--color-fg-subtle)]">null</span>;
  }
  return <span>{String(value)}</span>;
}

function JsonNode({ label, value, depth, defaultExpandDepth, linkResolver }: NodeProps) {
  const isContainer = Array.isArray(value) || isPlainObject(value);
  const [open, setOpen] = useState(depth < defaultExpandDepth);

  if (!isContainer) {
    return (
      <div className="flex gap-2 font-mono text-sm leading-relaxed">
        <span className="text-[var(--color-fg-subtle)]">{label}:</span>
        <PrimitiveValue label={label} value={value} linkResolver={linkResolver} />
      </div>
    );
  }

  const entries = Array.isArray(value)
    ? value.map((v, i) => [String(i), v] as const)
    : Object.entries(value);

  return (
    <div className="font-mono text-sm leading-relaxed">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex items-center gap-1 text-left text-[var(--color-fg-subtle)] hover:text-[var(--color-fg)] focus-visible:rounded focus-visible:outline-2 focus-visible:outline-[var(--color-accent)]"
      >
        <span aria-hidden className="inline-block w-3 text-xs">
          {open ? "▾" : "▸"}
        </span>
        <span>{label}</span>
        <span className="text-[var(--color-fg-subtle)]">{Array.isArray(value) ? " [" : " {"}</span>
        {!open && (
          <span className="ml-1 text-xs text-[var(--color-fg-subtle)]">
            {summarize(value)}
            {Array.isArray(value) ? "]" : "}"}
          </span>
        )}
      </button>
      {open && (
        <div className="ml-4 border-l border-[var(--color-border)] pl-3">
          {entries.length === 0 ? (
            <div className="text-[var(--color-fg-subtle)]">(empty)</div>
          ) : (
            entries.map(([k, v]) => (
              <JsonNode
                key={k}
                label={k}
                value={v}
                depth={depth + 1}
                defaultExpandDepth={defaultExpandDepth}
                linkResolver={linkResolver}
              />
            ))
          )}
          <div className="text-[var(--color-fg-subtle)]">{Array.isArray(value) ? "]" : "}"}</div>
        </div>
      )}
    </div>
  );
}

export function JsonTree({
  data,
  rootLabel = "$",
  linkResolver = defaultLinkResolver,
  defaultExpandDepth = 2,
  className,
}: JsonTreeProps) {
  return (
    <div
      className={cn(
        "rounded-md border border-[var(--color-border)] bg-[var(--color-bg-subtle)] p-3",
        className,
      )}
    >
      <JsonNode
        label={rootLabel}
        value={data}
        depth={0}
        defaultExpandDepth={defaultExpandDepth}
        linkResolver={linkResolver}
      />
    </div>
  );
}
