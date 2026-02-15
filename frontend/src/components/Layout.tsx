import type { ReactNode } from "react";

export type View = "generate" | "progress" | "list" | "detail";

interface LayoutProps {
  currentView: View;
  onNavigate: (view: View) => void;
  children: ReactNode;
}

const NAV_ITEMS: { view: View; label: string }[] = [
  { view: "generate", label: "Generate" },
  { view: "list", label: "Projects" },
];

export function Layout({ currentView, onNavigate, children }: LayoutProps) {
  return (
    <div className="min-h-screen">
      <header className="border-b border-gray-800 bg-gray-900/80 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-5xl items-center justify-between px-4">
          <button
            onClick={() => onNavigate("generate")}
            className="text-lg font-bold tracking-tight text-white"
          >
            vidpipe
          </button>
          <nav className="flex gap-1">
            {NAV_ITEMS.map(({ view, label }) => (
              <button
                key={view}
                onClick={() => onNavigate(view)}
                className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                  currentView === view
                    ? "bg-gray-800 text-white"
                    : "text-gray-400 hover:text-gray-200"
                }`}
              >
                {label}
              </button>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-5xl px-4 py-6">{children}</main>
    </div>
  );
}
