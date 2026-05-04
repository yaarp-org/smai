import { useEffect, useState } from "react";

import { cn } from "@/lib/utils";

import {
  DARK_THEME,
  LIGHT_THEME,
  SUPPORTED_LANGUAGES,
  type SupportedLanguage,
  getShikiHighlighter,
} from "./shiki-singleton";

export interface CodeViewerProps {
  code: string;
  language?: string | null;
  className?: string;
}

function usePrefersDark(): boolean {
  const [prefers, setPrefers] = useState(() => {
    if (typeof window === "undefined" || !window.matchMedia) return false;
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  });
  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = (e: MediaQueryListEvent) => setPrefers(e.matches);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);
  return prefers;
}

function isSupportedLanguage(lang: string | null | undefined): lang is SupportedLanguage {
  if (!lang) return false;
  return (SUPPORTED_LANGUAGES as readonly string[]).includes(lang);
}

export function CodeViewer({ code, language, className }: CodeViewerProps) {
  const prefersDark = usePrefersDark();
  const theme = prefersDark ? DARK_THEME : LIGHT_THEME;
  const lang = isSupportedLanguage(language) ? language : null;
  const [html, setHtml] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    if (!lang) {
      setHtml(null);
      return;
    }
    setFailed(false);
    setHtml(null);
    getShikiHighlighter()
      .then((hl) => {
        if (cancelled) return;
        const rendered = hl.codeToHtml(code, { lang, theme });
        setHtml(rendered);
      })
      .catch(() => {
        if (cancelled) return;
        setFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, [code, lang, theme]);

  if (lang && html && !failed) {
    return (
      <div
        // codeToHtml returns trusted output from a local Shiki bundle (no user-controlled HTML
        // is interpolated into the grammar input), so dangerouslySetInnerHTML is the documented
        // Shiki render path here.
        className={cn(
          "shiki-host overflow-x-auto rounded-md border border-[var(--color-border)] bg-[var(--color-bg-subtle)] p-3 text-sm leading-relaxed",
          "[&_pre]:!m-0 [&_pre]:!bg-transparent [&_pre]:!p-0",
          className,
        )}
        dangerouslySetInnerHTML={{ __html: html }}
      />
    );
  }

  return (
    <pre
      className={cn(
        "overflow-x-auto rounded-md border border-[var(--color-border)] bg-[var(--color-bg-subtle)] p-3 font-mono text-sm leading-relaxed text-[var(--color-fg)]",
        className,
      )}
    >
      <code>{code}</code>
    </pre>
  );
}
