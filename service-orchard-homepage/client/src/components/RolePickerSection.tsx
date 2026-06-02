import { roleCards } from "@/lib/data";
import { ArrowRight } from "lucide-react";
import { toast } from "sonner";

export default function RolePickerSection() {
  return (
    <section id="roles" className="py-20 md:py-28 bg-sage-light/40">
      <div className="container">
        <div className="max-w-2xl mb-12">
          <h2 className="font-display text-3xl md:text-4xl font-bold text-foreground mb-4">
            Pick your job. See what can be automated.
          </h2>
          <p className="text-lg text-muted-foreground leading-relaxed">
            Service Orchard is organized around real jobs, not abstract automation categories. Start with your role and we'll show you the repetitive tasks that can become AI skills.
          </p>
        </div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
          {roleCards.map((role, i) => (
            <div
              key={role.id}
              className="group bg-card rounded-xl p-6 border border-border/60 shadow-sm hover:shadow-md hover:border-forest/30 transition-all duration-200 hover:-translate-y-0.5"
              style={{ animationDelay: `${i * 50}ms` }}
            >
              <div className="flex items-center gap-2 mb-3">
                <span className="inline-block px-2 py-0.5 text-xs font-medium rounded-full bg-sage-light text-forest">
                  {role.industry}
                </span>
              </div>
              <h3 className="font-display text-lg font-semibold text-foreground mb-2">
                {role.title}
              </h3>
              <p className="text-sm text-muted-foreground leading-relaxed mb-4">
                {role.pain}
              </p>
              <button
                onClick={() => toast("Feature coming soon — role detail pages are in development.")}
                className="inline-flex items-center gap-1.5 text-sm font-semibold text-forest hover:text-forest-light transition-colors duration-150 group-hover:gap-2.5"
              >
                {role.cta}
                <ArrowRight className="w-3.5 h-3.5 transition-transform duration-150 group-hover:translate-x-0.5" />
              </button>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
