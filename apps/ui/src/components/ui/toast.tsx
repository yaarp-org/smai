import { Toaster as SonnerToaster } from "sonner";

// shadcn/ui's "toast" primitive uses Sonner under the hood. The SPA mounts a
// single <Toaster /> from this module in the root layout; per-call usage is
// `import { toast } from "sonner"` at the call site.
export function Toaster() {
  return (
    <SonnerToaster
      richColors
      position="top-right"
      toastOptions={{
        classNames: {
          toast: "border bg-background text-foreground",
        },
      }}
    />
  );
}

export { toast } from "sonner";
