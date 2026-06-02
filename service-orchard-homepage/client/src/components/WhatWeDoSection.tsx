import { whatWeDoPoints } from "@/lib/data";
import { FileText, ListChecks, Layers, BarChart3, CheckSquare } from "lucide-react";

const iconMap: Record<string, React.ElementType> = {
  FileText,
  ListChecks,
  Layers,
  BarChart3,
  CheckSquare,
};

export default function WhatWeDoSection() {
  return (
    <section id="what-we-do" className="py-20 md:py-28">
      <div className="container">
        <div className="max-w-3xl mx-auto text-center mb-14">
          <h2 className="font-display text-3xl md:text-4xl font-bold text-foreground mb-5">
            We build AI skills around the job you already have.
          </h2>
          <p className="text-lg text-muted-foreground leading-relaxed">
            Service Orchard studies the repetitive work inside your role and helps build small AI-powered skills for emails, follow-ups, scheduling, reports, notes, forms, CRM updates, document review, and daily admin work.
          </p>
          <p className="text-base text-muted-foreground mt-4">
            These are not generic chatbots. They are job-specific skills designed around the work you already do every day.
          </p>
        </div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-5 gap-6">
          {whatWeDoPoints.map((point) => {
            const Icon = iconMap[point.icon];
            return (
              <div
                key={point.title}
                className="text-center p-5 rounded-xl bg-card border border-border/40 hover:border-amber/40 hover:shadow-sm transition-all duration-200"
              >
                <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-sage-light text-forest mb-4">
                  <Icon className="w-5 h-5" />
                </div>
                <h3 className="font-display text-base font-semibold text-foreground mb-2">
                  {point.title}
                </h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {point.description}
                </p>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
