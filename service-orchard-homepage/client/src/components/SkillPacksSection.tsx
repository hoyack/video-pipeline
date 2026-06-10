import { skillPacks } from "@/lib/data";
import { Button } from "@/components/ui/button";
import { Check, ArrowRight } from "lucide-react";
import { toast } from "sonner";

export default function SkillPacksSection() {
  return (
    <section id="skill-packs" className="py-20 md:py-28">
      <div className="container">
        <div className="max-w-2xl mb-12">
          <h2 className="font-display text-3xl md:text-4xl font-bold text-foreground mb-4">
            Skill Packs for real workdays.
          </h2>
          <p className="text-lg text-muted-foreground leading-relaxed">
            A Skill Pack is a small bundle of AI skills designed around a specific role. Start with one repetitive part of the job, then grow from there.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {skillPacks.map((pack) => (
            <div
              key={pack.id}
              className="bg-card rounded-xl border border-border/60 p-6 md:p-8 hover:shadow-lg hover:border-forest/20 transition-all duration-200"
            >
              <h3 className="font-display text-xl font-bold text-foreground mb-2">
                {pack.title}
              </h3>
              <p className="text-sm text-muted-foreground mb-5 leading-relaxed">
                {pack.description}
              </p>

              <div className="space-y-2.5 mb-6">
                {pack.skills.map((skill) => (
                  <div key={skill} className="flex items-start gap-2.5">
                    <div className="mt-0.5 w-4 h-4 rounded-full bg-sage-light flex items-center justify-center shrink-0">
                      <Check className="w-2.5 h-2.5 text-forest" />
                    </div>
                    <span className="text-sm text-foreground/80">{skill}</span>
                  </div>
                ))}
              </div>

              <Button
                onClick={() => toast("Feature coming soon — scan pages are in development.")}
                className="bg-forest text-cream hover:bg-forest-light font-semibold transition-transform duration-150 active:scale-[0.97]"
              >
                {pack.cta}
                <ArrowRight className="w-4 h-4 ml-2" />
              </Button>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
