import { useEffect, useState } from "react";
import ReactDiffViewer from "react-diff-viewer-continued";

import { cn } from "@/lib/utils";

export interface DiffViewerProps {
  oldValue: string;
  newValue: string;
  oldTitle?: string;
  newTitle?: string;
  splitView?: boolean;
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

export function DiffViewer({
  oldValue,
  newValue,
  oldTitle,
  newTitle,
  splitView = true,
  className,
}: DiffViewerProps) {
  const prefersDark = usePrefersDark();
  return (
    <div
      className={cn(
        "overflow-hidden rounded-md border border-[var(--color-border)] text-sm",
        className,
      )}
    >
      <ReactDiffViewer
        oldValue={oldValue}
        newValue={newValue}
        leftTitle={oldTitle}
        rightTitle={newTitle}
        splitView={splitView}
        useDarkTheme={prefersDark}
      />
    </div>
  );
}
