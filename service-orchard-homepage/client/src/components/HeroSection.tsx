import { Button } from "@/components/ui/button";
import { ArrowRight, Sparkles } from "lucide-react";

const HERO_BG = "https://d2xsxph8kpxj0f.cloudfront.net/310419663029858924/dZZxW442BmUighmcfEtAyC/hero-bg-3MTcvBy6Pvj46ejCjt53wY.webp";
const MESSY = "https://d2xsxph8kpxj0f.cloudfront.net/310419663029858924/dZZxW442BmUighmcfEtAyC/hero-split-messy-3fUq3w64g3ytfqcYy97iZr.webp";
const CLEAN = "https://d2xsxph8kpxj0f.cloudfront.net/310419663029858924/dZZxW442BmUighmcfEtAyC/hero-split-clean-4CGUZKFwufNYob3pFRx89q.webp";

export default function HeroSection() {
  return (
    <section
      id="hero"
      className="relative pt-24 pb-16 md:pt-32 md:pb-24 overflow-hidden"
    >
      {/* Background */}
      <div className="absolute inset-0 -z-10">
        <img
          src={HERO_BG}
          alt=""
          className="w-full h-full object-cover opacity-30"
        />
        <div className="absolute inset-0 bg-gradient-to-b from-background/60 via-background/80 to-background" />
      </div>

      <div className="container">
        <div className="grid lg:grid-cols-2 gap-12 lg:gap-16 items-center">
          {/* Left: Copy */}
          <div className="max-w-xl">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-sage-light text-forest text-sm font-medium mb-6">
              <Sparkles className="w-4 h-4" />
              AI skills for real jobs
            </div>

            <h1 className="font-display text-4xl md:text-5xl lg:text-6xl font-bold text-foreground leading-[1.1] tracking-tight mb-6">
              Automate the boring parts of your job.
            </h1>

            <p className="text-lg md:text-xl text-muted-foreground leading-relaxed mb-8">
              Service Orchard builds AI skills for real roles in HVAC, plumbing, electrical, construction, real estate, and homebuilding. Pick your job, show us what slows you down, and we'll help you turn repetitive work into AI-powered output.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 mb-6">
              <Button
                asChild
                size="lg"
                className="bg-forest text-cream hover:bg-forest-light font-semibold text-base px-6 shadow-lg shadow-forest/20 transition-transform duration-150 active:scale-[0.97]"
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
                className="border-forest/30 text-forest hover:bg-sage-light font-medium text-base"
              >
                <a href="#roles">Browse Jobs We Can Upgrade</a>
              </Button>
            </div>

            <p className="text-sm text-muted-foreground">
              No generic chatbot pitch. No giant software rollout. Just practical AI skills built around the work you already do.
            </p>
          </div>

          {/* Right: Split Visual */}
          <div className="relative hidden lg:block">
            <div className="grid grid-cols-2 gap-4">
              {/* Before */}
              <div className="relative group">
                <div className="absolute -top-3 left-3 bg-destructive/90 text-white text-xs font-semibold px-2.5 py-1 rounded-full z-10">
                  Before
                </div>
                <div className="rounded-xl overflow-hidden shadow-xl border border-border/50 transition-transform duration-300 group-hover:scale-[1.02]">
                  <img
                    src={MESSY}
                    alt="Chaotic workday with scattered tasks"
                    className="w-full h-80 object-cover object-top"
                  />
                </div>
              </div>
              {/* After */}
              <div className="relative group">
                <div className="absolute -top-3 left-3 bg-forest/90 text-cream text-xs font-semibold px-2.5 py-1 rounded-full z-10">
                  After
                </div>
                <div className="rounded-xl overflow-hidden shadow-xl border border-border/50 transition-transform duration-300 group-hover:scale-[1.02]">
                  <img
                    src={CLEAN}
                    alt="Organized AI-assisted workspace"
                    className="w-full h-80 object-cover object-top"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
