import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export interface PlaceholderPageProps {
  title: string;
  description: string;
  upcomingTask: string;
}

export function PlaceholderPage({ title, description, upcomingTask }: PlaceholderPageProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-[var(--color-fg-subtle)]">
          Real content lands in <span className="font-mono">{upcomingTask}</span>.
        </p>
      </CardContent>
    </Card>
  );
}
