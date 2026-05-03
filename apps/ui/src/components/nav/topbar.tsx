import { Link } from "@tanstack/react-router";

import { cn } from "@/lib/utils";

// IA per 13-frontend.md §11.1: 5 nav items map to the 5 primary resources from
// 11-api.md §3.4. Proposals leads (DEC-032). The smai logo links to "/".
const NAV_ITEMS = [
  { to: "/proposals", label: "Proposals" },
  { to: "/papers", label: "Papers" },
  { to: "/comparison-groups", label: "Comparison Groups" },
  { to: "/runs", label: "Runs" },
  { to: "/system", label: "System" },
] as const;

export function Topbar() {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-[var(--color-border)] bg-[var(--color-bg)]">
      <div className="mx-auto flex h-14 max-w-7xl items-center gap-6 px-4">
        <Link to="/" className="text-base font-bold tracking-tight text-[var(--color-fg)]">
          smai
        </Link>
        <nav className="flex items-center gap-1 text-sm">
          {NAV_ITEMS.map((item) => (
            <Link
              key={item.to}
              to={item.to}
              className={cn(
                "rounded-md px-3 py-1.5 text-[var(--color-fg-subtle)] transition-colors hover:bg-[var(--color-bg-subtle)] hover:text-[var(--color-fg)]",
              )}
              activeProps={{
                className: "bg-[var(--color-bg-subtle)] text-[var(--color-fg)] font-medium",
              }}
            >
              {item.label}
            </Link>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-2">
          {/* cmd-K palette placeholder per OQ6 RESOLVED 2026-05-03 (deferred).
              The kbd renders as a visual affordance only; 4.M4-or-later wires
              the actual command palette behavior. */}
          <kbd
            aria-hidden
            className="hidden h-7 items-center gap-1 rounded-md border border-[var(--color-border)] bg-[var(--color-bg-subtle)] px-2 text-xs font-medium text-[var(--color-fg-subtle)] select-none sm:inline-flex"
          >
            <span className="text-[10px]">⌘</span>K
          </kbd>
        </div>
      </div>
    </header>
  );
}
