import { ErrorBoundary } from "./ErrorBoundary";

export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="mx-auto max-w-7xl p-6">
      <ErrorBoundary>
        {children}
      </ErrorBoundary>
    </div>
  );
}
