import { Outlet, createRootRoute } from "@tanstack/react-router";

import { Topbar } from "@/components/nav/topbar";
import { Toaster } from "@/components/ui/toast";
import { TooltipProvider } from "@/components/ui/tooltip";

export const Route = createRootRoute({
  component: RootLayout,
});

function RootLayout() {
  return (
    <TooltipProvider>
      <div className="flex min-h-screen flex-col bg-[var(--color-bg)] text-[var(--color-fg)]">
        <Topbar />
        <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6">
          <Outlet />
        </main>
      </div>
      <Toaster />
    </TooltipProvider>
  );
}
