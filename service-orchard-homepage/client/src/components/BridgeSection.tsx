import { Button } from "@/components/ui/button";
import { User, Building2, ArrowRight } from "lucide-react";
import { toast } from "sonner";

export default function BridgeSection() {
  return (
    <section id="bridge" className="py-20 md:py-28 bg-sage-light/40">
      <div className="container">
        <div className="max-w-3xl mx-auto text-center mb-14">
          <h2 className="font-display text-3xl md:text-4xl font-bold text-foreground">
            Built for the person doing the work. Useful to the person managing it.
          </h2>
        </div>

        <div className="grid md:grid-cols-2 gap-6 max-w-4xl mx-auto">
          {/* For the worker */}
          <div className="bg-card rounded-xl border border-border/60 p-7 md:p-8">
            <div className="inline-flex items-center justify-center w-11 h-11 rounded-full bg-forest/10 text-forest mb-5">
              <User className="w-5 h-5" />
            </div>
            <h3 className="font-display text-xl font-semibold text-foreground mb-3">
              For the person doing the job
            </h3>
            <p className="text-muted-foreground leading-relaxed mb-4">
              Service Orchard helps you get more done, respond faster, stay organized, and stop wasting your best hours on repetitive admin work.
            </p>
            <p className="text-sm font-medium text-forest">
              You keep the judgment. You get help with the busywork.
            </p>
          </div>

          {/* For the manager */}
          <div className="bg-card rounded-xl border border-border/60 p-7 md:p-8">
            <div className="inline-flex items-center justify-center w-11 h-11 rounded-full bg-amber/15 text-amber-dark mb-5">
              <Building2 className="w-5 h-5" />
            </div>
            <h3 className="font-display text-xl font-semibold text-foreground mb-3">
              For the person managing the job
            </h3>
            <p className="text-muted-foreground leading-relaxed mb-4">
              Service Orchard can help your best employees handle more volume, reduce dropped balls, and turn tribal knowledge into repeatable workflows.
            </p>
            <p className="text-sm text-muted-foreground">
              When the workflow needs to be fully managed, Thunderstaff can take over the process.
            </p>
          </div>
        </div>

        <div className="text-center mt-10">
          <Button
            variant="outline"
            onClick={() => toast("Feature coming soon — Thunderstaff page is in development.")}
            className="border-forest/30 text-forest hover:bg-sage-light font-medium"
          >
            Want this managed for the whole business? Ask about Thunderstaff
            <ArrowRight className="w-4 h-4 ml-2" />
          </Button>
        </div>
      </div>
    </section>
  );
}
