import { offerLadder } from "@/lib/data";
import { Button } from "@/components/ui/button";
import { ArrowRight } from "lucide-react";
import { toast } from "sonner";

const GROWTH_IMG = "https://d2xsxph8kpxj0f.cloudfront.net/310419663029858924/dZZxW442BmUighmcfEtAyC/growth-illustration-Z5ZuBEK8gzZW9u6cKZajzY.webp";

export default function OfferLadderSection() {
  return (
    <section id="offer-ladder" className="py-20 md:py-28">
      <div className="container">
        <div className="grid lg:grid-cols-2 gap-12 items-center mb-14">
          <div>
            <h2 className="font-display text-3xl md:text-4xl font-bold text-foreground mb-4">
              Start with one job. Grow into a system.
            </h2>
            <p className="text-lg text-muted-foreground leading-relaxed">
              Service Orchard upgrades the worker. Thunderstaff manages the workflow. Hoyack builds the system.
            </p>
          </div>
          <div className="hidden lg:block">
            <img
              src={GROWTH_IMG}
              alt="Growth progression from seed to full tree"
              className="w-full max-w-md ml-auto rounded-xl"
            />
          </div>
        </div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {offerLadder.map((offer, i) => (
            <div
              key={offer.step}
              className="relative bg-card rounded-xl border border-border/60 p-5 hover:shadow-md hover:border-forest/20 transition-all duration-200 group"
            >
              <div className="flex items-center gap-2 mb-3">
                <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-forest text-cream text-xs font-bold">
                  {offer.step}
                </span>
                {i < offerLadder.length - 1 && (
                  <div className="hidden lg:block absolute -right-2.5 top-1/2 -translate-y-1/2 z-10">
                    <ArrowRight className="w-4 h-4 text-border" />
                  </div>
                )}
              </div>
              <h3 className="font-display text-sm font-semibold text-foreground mb-2 leading-tight">
                {offer.title}
              </h3>
              <p className="text-xs text-muted-foreground leading-relaxed mb-4">
                {offer.description}
              </p>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  if (offer.step === 1) {
                    document.getElementById("scan")?.scrollIntoView({ behavior: "smooth" });
                  } else {
                    toast("Feature coming soon.");
                  }
                }}
                className="text-xs border-forest/20 text-forest hover:bg-sage-light font-medium w-full"
              >
                {offer.cta}
              </Button>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
