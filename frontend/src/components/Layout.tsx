import type { ReactNode } from "react";
import { Link, useLocation } from "wouter";

interface LayoutProps {
  children: ReactNode;
}

const NAV_ITEMS: { href: string; label: string; match: (path: string) => boolean }[] = [
  { href: "/dashboard", label: "Dashboard", match: (path) => path === "/dashboard" },
  {
    href: "/productions",
    label: "Productions",
    match: (path) => path.startsWith("/productions"),
  },
  {
    href: "/",
    label: "Scenes",
    match: (path) =>
      path === "/" ||
      path === "/generate" ||
      path.startsWith("/scenes/"),
  },
  { href: "/manifests", label: "Manifests", match: (path) => path.startsWith("/manifests") },
  { href: "/settings", label: "Settings", match: (path) => path === "/settings" },
];

export function Layout({ children }: LayoutProps) {
  const [location] = useLocation();

  return (
    <div className="min-h-screen">
      <header className="border-b border-gray-800 bg-gray-900/80 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-5xl items-center justify-between px-4">
          <Link
            href="/"
            className="text-lg font-bold tracking-tight text-white"
          >
            vidpipe
          </Link>
          <nav className="flex gap-1">
            {NAV_ITEMS.map(({ href, label, match }) => {
              const isActive = match(location);
              return (
                <Link
                  key={href}
                  href={href}
                  className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                    isActive
                      ? "bg-gray-800 text-white"
                      : "text-gray-400 hover:text-gray-200"
                  }`}
                >
                  {label}
                </Link>
              );
            })}
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-5xl px-4 py-6">{children}</main>
    </div>
  );
}
