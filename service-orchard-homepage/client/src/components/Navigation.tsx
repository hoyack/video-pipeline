import { useState } from "react";
import { Button } from "@/components/ui/button";
import { navLinks } from "@/lib/data";
import { Menu, X, TreePine } from "lucide-react";

export default function Navigation() {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-background/90 backdrop-blur-md border-b border-border/50">
      <div className="container flex items-center justify-between h-16">
        {/* Logo */}
        <a href="#" className="flex items-center gap-2 group">
          <TreePine className="w-7 h-7 text-forest" />
          <span className="font-display font-bold text-xl text-forest tracking-tight">
            Service Orchard
          </span>
        </a>

        {/* Desktop Nav */}
        <div className="hidden md:flex items-center gap-8">
          {navLinks.map((link) => (
            <a
              key={link.label}
              href={link.href}
              className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors duration-150"
            >
              {link.label}
            </a>
          ))}
          <Button
            asChild
            className="bg-forest text-cream hover:bg-forest-light font-semibold shadow-sm"
          >
            <a href="#scan">Start Free Scan</a>
          </Button>
        </div>

        {/* Mobile Toggle */}
        <button
          className="md:hidden p-2 text-foreground"
          onClick={() => setMobileOpen(!mobileOpen)}
          aria-label="Toggle menu"
        >
          {mobileOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
        </button>
      </div>

      {/* Mobile Menu */}
      {mobileOpen && (
        <div className="md:hidden bg-background border-b border-border animate-fade-in">
          <div className="container py-4 flex flex-col gap-4">
            {navLinks.map((link) => (
              <a
                key={link.label}
                href={link.href}
                className="text-base font-medium text-foreground py-2"
                onClick={() => setMobileOpen(false)}
              >
                {link.label}
              </a>
            ))}
            <Button
              asChild
              className="bg-forest text-cream hover:bg-forest-light font-semibold w-full"
            >
              <a href="#scan" onClick={() => setMobileOpen(false)}>Start Free Scan</a>
            </Button>
          </div>
        </div>
      )}
    </nav>
  );
}
