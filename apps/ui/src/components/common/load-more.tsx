import { Button } from "@/components/ui/button";

// Cursor-based pagination button per 13-frontend.md §11.2 acceptance and
// 11-api.md §5.1.1: opaque cursor round-trips as ?cursor=. The list page owns
// the URL search-param state; this component just exposes the current cursor +
// onClick navigator.

export interface LoadMoreProps {
  nextCursor: string | null | undefined;
  onLoadMore: (cursor: string) => void;
  loading?: boolean;
}

export function LoadMore({ nextCursor, onLoadMore, loading }: LoadMoreProps) {
  if (!nextCursor) return null;
  return (
    <div className="flex justify-center pt-2">
      <Button variant="outline" size="sm" onClick={() => onLoadMore(nextCursor)} disabled={loading}>
        Load more
      </Button>
    </div>
  );
}
