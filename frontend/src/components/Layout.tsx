import type { ReactNode } from "react";

export type View = "generate" | "progress" | "list" | "detail" | "dashboard" | "manifests" | "manifest-creator" | "settings";

interface LayoutProps {
  currentView: View;
  onNavigate: (view: View) => void;
  children: ReactNode;
}

const NAV_ITEMS: { view: View; label: string; activeFor?: View[] }[] = [
  { view: "list", label: "Projects", activeFor: ["generate"] },
  { view: "manifests", label: "Manifests" },
  { view: "dashboard", label: "Dashboard" },
  { view: "settings", label: "Settings" },
];

export function Layout({ currentView, onNavigate, children }: LayoutProps) {
  return (
    <div className="min-h-screen">
      <header className="border-b border-gray-800 bg-gray-900/80 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-5xl items-center justify-between px-4">
          <button
            onClick={() => onNavigate("list")}
            className="text-lg font-bold tracking-tight text-white"
          >
            vidpipe
          </button>
          <nav className="flex gap-1">
            {NAV_ITEMS.map(({ view, label, activeFor }) => {
              const isActive = currentView === view || (activeFor?.includes(currentView) ?? false);
              return (
                <button
                  key={view}
                  onClick={() => onNavigate(view)}
                  className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                    isActive
                      ? "bg-gray-800 text-white"
                      : "text-gray-400 hover:text-gray-200"
                  }`}
                >
                  {label}
                </button>
              );
            })}
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-5xl px-4 py-6">{children}</main>
    </div>
  );
}
