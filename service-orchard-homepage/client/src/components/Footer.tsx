import { footerColumns } from "@/lib/data";
import { TreePine } from "lucide-react";

export default function Footer() {
  return (
    <footer className="py-14 md:py-20 border-t border-border/60">
      <div className="container">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-12">
          {footerColumns.map((col) => (
            <div key={col.title}>
              <h4 className="font-display text-sm font-semibold text-foreground mb-4">
                {col.title}
              </h4>
              <ul className="space-y-2.5">
                {col.links.map((link) => (
                  <li key={link.label}>
                    <a
                      href={link.href}
                      className="text-sm text-muted-foreground hover:text-foreground transition-colors duration-150"
                    >
                      {link.label}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="flex flex-col md:flex-row items-center justify-between gap-4 pt-8 border-t border-border/40">
          <div className="flex items-center gap-2">
            <TreePine className="w-5 h-5 text-forest" />
            <span className="font-display font-semibold text-sm text-forest">Service Orchard</span>
          </div>
          <p className="text-xs text-muted-foreground">
            &copy; {new Date().getFullYear()} Service Orchard. Part of the Hoyack automation stack.
          </p>
        </div>
      </div>
    </footer>
  );
}
