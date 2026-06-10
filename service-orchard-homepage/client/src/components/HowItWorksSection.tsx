import { howItWorksSteps } from "@/lib/data";

export default function HowItWorksSection() {
  return (
    <section id="how-it-works" className="py-20 md:py-28 bg-sage-light/40">
      <div className="container">
        <div className="max-w-2xl mx-auto text-center mb-14">
          <h2 className="font-display text-3xl md:text-4xl font-bold text-foreground mb-4">
            How Service Orchard works.
          </h2>
        </div>

        <div className="max-w-3xl mx-auto">
          <div className="relative">
            {/* Vertical line */}
            <div className="absolute left-5 top-0 bottom-0 w-px bg-border hidden md:block" />

            <div className="space-y-8">
              {howItWorksSteps.map((step, i) => (
                <div key={step.step} className="flex gap-5 md:gap-8 items-start group">
                  {/* Step number */}
                  <div className="relative z-10 shrink-0 w-10 h-10 rounded-full bg-forest text-cream flex items-center justify-center font-display font-bold text-sm shadow-md">
                    {step.step}
                  </div>

                  {/* Content */}
                  <div className="pb-2">
                    <h3 className="font-display text-lg font-semibold text-foreground mb-1">
                      {step.title}
                    </h3>
                    <p className="text-muted-foreground leading-relaxed">
                      {step.description}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
