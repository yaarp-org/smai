import type { BundledLanguage, BundledTheme, Highlighter } from "shiki";

// Shared Shiki highlighter so CodeViewer + MarkdownViewer don't each pay the
// engine init cost. The bundled-language allowlist below maps onto the file
// extensions ArtifactFrame dispatches to (Python harness/technique source,
// arXiv .tex, build/config files). Keep this list narrow so Vite tree-shakes
// the rest of Shiki's grammar bundle out of the main chunk.
export const SUPPORTED_LANGUAGES = [
  "python",
  "typescript",
  "javascript",
  "tsx",
  "jsx",
  "json",
  "yaml",
  "toml",
  "bash",
  "shellscript",
  "latex",
  "markdown",
] as const satisfies readonly BundledLanguage[];

export type SupportedLanguage = (typeof SUPPORTED_LANGUAGES)[number];

export const LIGHT_THEME: BundledTheme = "github-light";
export const DARK_THEME: BundledTheme = "github-dark";

let highlighterPromise: Promise<Highlighter> | null = null;

export function getShikiHighlighter(): Promise<Highlighter> {
  if (!highlighterPromise) {
    highlighterPromise = (async () => {
      const { createHighlighter } = await import("shiki");
      return createHighlighter({
        themes: [LIGHT_THEME, DARK_THEME],
        langs: [...SUPPORTED_LANGUAGES],
      });
    })();
  }
  return highlighterPromise;
}

const EXTENSION_TO_LANGUAGE: Record<string, SupportedLanguage> = {
  py: "python",
  ts: "typescript",
  tsx: "tsx",
  js: "javascript",
  jsx: "jsx",
  json: "json",
  yaml: "yaml",
  yml: "yaml",
  toml: "toml",
  sh: "bash",
  bash: "bash",
  tex: "latex",
  md: "markdown",
};

export function detectLanguage(path: string): SupportedLanguage | null {
  const dot = path.lastIndexOf(".");
  if (dot < 0) return null;
  const ext = path.slice(dot + 1).toLowerCase();
  return EXTENSION_TO_LANGUAGE[ext] ?? null;
}
