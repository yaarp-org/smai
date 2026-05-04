import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

import { cn } from "@/lib/utils";

import { CodeViewer } from "./code-viewer";

export interface MarkdownViewerProps {
  content: string;
  className?: string;
}

// Map react-markdown's <code> output onto CodeViewer for fenced blocks; inline
// code falls through to a plain monospace span. Sharing CodeViewer keeps the
// Shiki highlighter singleton from being instantiated twice. We sidestep
// rehype-shiki@0.0.9 here because it is pinned to shiki@0.1.x and is not
// compatible with the shiki@4 instance the rest of the app loads — see the
// status note in the implementation_plan §3 task entry for 4.M6.
const COMPONENTS: Components = {
  code({ className, children, ...rest }) {
    const inline = !/language-/.test(className ?? "");
    if (inline) {
      return (
        <code
          className="rounded bg-[var(--color-bg-subtle)] px-1 py-0.5 font-mono text-[0.875em] text-[var(--color-fg)]"
          {...rest}
        >
          {children}
        </code>
      );
    }
    const match = /language-(\w+)/.exec(className ?? "");
    const language = match?.[1] ?? null;
    const code = String(children).replace(/\n$/, "");
    return <CodeViewer code={code} language={language} className="my-3" />;
  },
};

export function MarkdownViewer({ content, className }: MarkdownViewerProps) {
  return (
    <div
      className={cn(
        "prose-smai max-w-none text-[var(--color-fg)]",
        "[&>*]:my-3 [&>*:first-child]:mt-0 [&>*:last-child]:mb-0",
        "[&_h1]:text-2xl [&_h1]:font-semibold [&_h1]:tracking-tight",
        "[&_h2]:text-xl [&_h2]:font-semibold [&_h2]:tracking-tight",
        "[&_h3]:text-lg [&_h3]:font-semibold",
        "[&_p]:leading-7",
        "[&_a]:text-[var(--color-accent)] [&_a]:underline-offset-2 hover:[&_a]:underline",
        "[&_ul]:list-disc [&_ul]:pl-6",
        "[&_ol]:list-decimal [&_ol]:pl-6",
        "[&_blockquote]:border-l-2 [&_blockquote]:border-[var(--color-border)] [&_blockquote]:pl-3 [&_blockquote]:text-[var(--color-fg-subtle)]",
        "[&_table]:w-full [&_table]:border-collapse",
        "[&_th]:border [&_th]:border-[var(--color-border)] [&_th]:px-2 [&_th]:py-1 [&_th]:text-left",
        "[&_td]:border [&_td]:border-[var(--color-border)] [&_td]:px-2 [&_td]:py-1",
        "[&_hr]:border-[var(--color-border)]",
        className,
      )}
    >
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={COMPONENTS}>
        {content}
      </ReactMarkdown>
    </div>
  );
}
