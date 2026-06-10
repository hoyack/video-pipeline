import { Button } from "@/components/ui/button";
import { ArrowRight } from "lucide-react";

export default function FinalCTASection() {
  return (
    <section className="py-20 md:py-28 bg-forest relative overflow-hidden">
      {/* Decorative gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-forest via-forest to-forest-light opacity-80" />
      <div className="absolute top-0 right-0 w-96 h-96 rounded-full bg-amber/10 blur-3xl" />
      <div className="absolute bottom-0 left-0 w-64 h-64 rounded-full bg-sage/10 blur-2xl" />

      <div className="container relative z-10 text-center">
        <h2 className="font-display text-3xl md:text-4xl lg:text-5xl font-bold text-cream mb-5 max-w-2xl mx-auto">
          Your job is full of automatable work.
        </h2>
        <p className="text-lg text-cream/80 mb-8 max-w-lg mx-auto">
          Let's find the first task Service Orchard can take off your plate.
        </p>

        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Button
            asChild
            size="lg"
            className="bg-amber text-foreground hover:bg-amber-dark font-semibold text-base px-8 shadow-lg transition-transform duration-150 active:scale-[0.97]"
          >
            <a href="#scan">
              Start My Free Job Automation Scan
              <ArrowRight className="w-4 h-4 ml-2" />
            </a>
          </Button>
          <Button
            asChild
            variant="outline"
            size="lg"
            className="border-cream/30 text-cream hover:bg-white/10 font-medium text-base"
          >
            <a href="#roles">Browse Jobs We Can Upgrade</a>
          </Button>
        </div>
      </div>
    </section>
  );
}
