import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { ArrowRight, CheckCircle2 } from "lucide-react";

const SCAN_BG = "https://d2xsxph8kpxj0f.cloudfront.net/310419663029858924/dZZxW442BmUighmcfEtAyC/scan-section-bg-GosiZaykBkXVtHicnar2A9.webp";

export default function ScanSection() {
  const [submitted, setSubmitted] = useState(false);
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    company: "",
    jobTitle: "",
    industry: "",
    tools: "",
    repetitiveTasks: "",
    hatedTask: "",
    forWhom: "myself",
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Track conversion event
    if (typeof window !== "undefined" && (window as any).umami) {
      (window as any).umami.track("scan_form_submit", formData);
    }
    setSubmitted(true);
  };

  const handleChange = (field: string, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  if (submitted) {
    return (
      <section id="scan" className="py-20 md:py-28 relative overflow-hidden">
        <div className="absolute inset-0 -z-10 opacity-10">
          <img src={SCAN_BG} alt="" className="w-full h-full object-cover" />
        </div>
        <div className="container">
          <div className="max-w-lg mx-auto text-center">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-forest/10 text-forest mb-6">
              <CheckCircle2 className="w-8 h-8" />
            </div>
            <h2 className="font-display text-2xl md:text-3xl font-bold text-foreground mb-4">
              Thanks. We received your Job Automation Scan.
            </h2>
            <p className="text-muted-foreground leading-relaxed">
              We'll review your role and repetitive tasks, then identify the first few AI skills that could help.
            </p>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section id="scan" className="py-20 md:py-28 relative overflow-hidden">
      <div className="absolute inset-0 -z-10 opacity-8">
        <img src={SCAN_BG} alt="" className="w-full h-full object-cover" />
      </div>

      <div className="container">
        <div className="grid lg:grid-cols-2 gap-12 lg:gap-16 items-start">
          {/* Left: Copy */}
          <div className="max-w-lg">
            <h2 className="font-display text-3xl md:text-4xl font-bold text-foreground mb-5">
              Find out what parts of your job can be automated.
            </h2>
            <p className="text-lg text-muted-foreground leading-relaxed mb-6">
              Tell us your job title, your industry, the tools you use, and the tasks that eat up your day. We'll identify practical AI skills that could help you move faster, reduce repetitive work, and create cleaner output.
            </p>
            <div className="hidden lg:block">
              <div className="p-5 rounded-xl bg-sage-light/60 border border-forest/10">
                <p className="text-sm font-medium text-forest mb-2">What you'll get:</p>
                <ul className="space-y-2 text-sm text-foreground/70">
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-forest shrink-0" />
                    A list of automatable tasks in your role
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-forest shrink-0" />
                    Recommended first AI skills to build
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-forest shrink-0" />
                    A suggested Skill Pack for your job
                  </li>
                </ul>
              </div>
            </div>
          </div>

          {/* Right: Form */}
          <form
            onSubmit={handleSubmit}
            className="bg-card rounded-2xl border border-border/60 shadow-lg p-6 md:p-8 space-y-5"
          >
            <div className="grid sm:grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <Label htmlFor="name">Name</Label>
                <Input
                  id="name"
                  placeholder="Your name"
                  value={formData.name}
                  onChange={(e) => handleChange("name", e.target.value)}
                  required
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="you@company.com"
                  value={formData.email}
                  onChange={(e) => handleChange("email", e.target.value)}
                  required
                />
              </div>
            </div>

            <div className="grid sm:grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <Label htmlFor="company">Company</Label>
                <Input
                  id="company"
                  placeholder="Company name"
                  value={formData.company}
                  onChange={(e) => handleChange("company", e.target.value)}
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="jobTitle">Job Title</Label>
                <Input
                  id="jobTitle"
                  placeholder="e.g. HVAC Dispatcher"
                  value={formData.jobTitle}
                  onChange={(e) => handleChange("jobTitle", e.target.value)}
                  required
                />
              </div>
            </div>

            <div className="grid sm:grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <Label htmlFor="industry">Industry</Label>
                <Input
                  id="industry"
                  placeholder="e.g. HVAC, Plumbing, Real Estate"
                  value={formData.industry}
                  onChange={(e) => handleChange("industry", e.target.value)}
                  required
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="tools">What tools do you use?</Label>
                <Input
                  id="tools"
                  placeholder="e.g. ServiceTitan, HubSpot, Excel"
                  value={formData.tools}
                  onChange={(e) => handleChange("tools", e.target.value)}
                />
              </div>
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="repetitiveTasks">What are your most repetitive tasks?</Label>
              <Textarea
                id="repetitiveTasks"
                placeholder="Describe the tasks you do over and over..."
                rows={3}
                value={formData.repetitiveTasks}
                onChange={(e) => handleChange("repetitiveTasks", e.target.value)}
                required
              />
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="hatedTask">What task do you hate doing most?</Label>
              <Input
                id="hatedTask"
                placeholder="The one thing you'd automate first"
                value={formData.hatedTask}
                onChange={(e) => handleChange("hatedTask", e.target.value)}
              />
            </div>

            <div className="space-y-1.5">
              <Label>Are you asking for yourself or your company?</Label>
              <div className="flex gap-4">
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <input
                    type="radio"
                    name="forWhom"
                    value="myself"
                    checked={formData.forWhom === "myself"}
                    onChange={(e) => handleChange("forWhom", e.target.value)}
                    className="accent-forest"
                  />
                  For myself
                </label>
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <input
                    type="radio"
                    name="forWhom"
                    value="company"
                    checked={formData.forWhom === "company"}
                    onChange={(e) => handleChange("forWhom", e.target.value)}
                    className="accent-forest"
                  />
                  For my company
                </label>
              </div>
            </div>

            <Button
              type="submit"
              size="lg"
              className="w-full bg-forest text-cream hover:bg-forest-light font-semibold text-base shadow-lg shadow-forest/20 transition-transform duration-150 active:scale-[0.97]"
            >
              Start My Free Job Automation Scan
              <ArrowRight className="w-4 h-4 ml-2" />
            </Button>

            <p className="text-xs text-muted-foreground text-center">
              Free. No credit card. We'll review your role and get back to you.
            </p>
          </form>
        </div>
      </div>
    </section>
  );
}
