import { automatableCategories } from "@/lib/data";
import { Mail, StickyNote, Calendar, Bell, FileStack, Database } from "lucide-react";

const iconMap: Record<string, React.ElementType> = {
  Mail,
  StickyNote,
  Calendar,
  Bell,
  FileStack,
  Database,
};

export default function AutomatableSection() {
  return (
    <section id="automatable" className="py-20 md:py-28 bg-forest relative overflow-hidden">
      {/* Subtle pattern overlay */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute inset-0" style={{
          backgroundImage: `radial-gradient(circle at 1px 1px, rgba(255,255,255,0.3) 1px, transparent 0)`,
          backgroundSize: '32px 32px'
        }} />
      </div>

      <div className="container relative z-10">
        <div className="max-w-3xl mb-12">
          <h2 className="font-display text-3xl md:text-4xl font-bold text-cream mb-5">
            Your job probably has 10 tasks we can automate.
          </h2>
          <p className="text-lg text-cream/80 leading-relaxed">
            Most roles are a mix of judgment, communication, coordination, and repetitive admin. Service Orchard focuses on the repetitive parts so you can spend more time on the work that actually needs you.
          </p>
        </div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
          {automatableCategories.map((cat) => {
            const Icon = iconMap[cat.icon];
            return (
              <div
                key={cat.title}
                className="p-6 rounded-xl bg-white/5 border border-white/10 backdrop-blur-sm hover:bg-white/10 transition-all duration-200"
              >
                <div className="inline-flex items-center justify-center w-10 h-10 rounded-lg bg-amber/20 text-amber mb-4">
                  <Icon className="w-5 h-5" />
                </div>
                <h3 className="font-display text-base font-semibold text-cream mb-2">
                  {cat.title}
                </h3>
                <p className="text-sm text-cream/70 leading-relaxed">
                  {cat.description}
                </p>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
