import { motion, useInView } from "framer-motion";
import { useRef } from "react";
import { Target, Users, Zap, Shield } from "lucide-react";

const reasons = [
  {
    icon: Target,
    title: "Precision Targeting",
    description:
      "We don't spray and pray. Our team uses refined lists, strategic scripts, and deep industry knowledge to reach the right decision-makers at the right time.",
  },
  {
    icon: Users,
    title: "Dedicated Teams",
    description:
      "Every client gets a dedicated team that becomes an extension of your business. No shared agents, no divided attention — just focused results.",
  },
  {
    icon: Zap,
    title: "Rapid Deployment",
    description:
      "Need agents yesterday? Our digital nomad model means we can scale your team up or down within days, not weeks. Agility is our advantage.",
  },
  {
    icon: Shield,
    title: "Restricted Industry Experts",
    description:
      "We thrive where others can't operate. Industries with advertising restrictions rely on us to build their pipeline through compliant outbound strategies.",
  },
];

export default function WhyUs() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-80px" });

  return (
    <section className="py-24 lg:py-32 bg-white">
      <div className="container" ref={ref}>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-16">
          {/* Left: Header */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={isInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6 }}
            className="lg:col-span-1"
          >
            <span className="text-[#E8722A] font-semibold text-sm uppercase tracking-wider mb-3 block">
              Why Swift Dialers
            </span>
            <h2 className="font-display text-3xl sm:text-4xl font-bold text-[#1A1F36] mb-5 leading-tight">
              The Unfair Advantage Your Business Needs
            </h2>
            <p className="text-gray-600 leading-relaxed">
              In a world of automated bots and impersonal outreach, we bring the human touch
              backed by professional discipline and cutting-edge processes.
            </p>
          </motion.div>

          {/* Right: Cards */}
          <div className="lg:col-span-2 grid grid-cols-1 sm:grid-cols-2 gap-6">
            {reasons.map((reason, index) => (
              <motion.div
                key={reason.title}
                initial={{ opacity: 0, y: 30 }}
                animate={isInView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.5, delay: 0.1 + index * 0.1 }}
                className="p-6 rounded-2xl border border-gray-100 hover:border-[#E8722A]/20 hover:shadow-lg hover:shadow-[#E8722A]/5 transition-all duration-300 group"
              >
                <div className="w-11 h-11 rounded-xl bg-[#1A1F36] flex items-center justify-center mb-4 group-hover:bg-[#E8722A] transition-colors duration-300">
                  <reason.icon size={20} className="text-white" />
                </div>
                <h3 className="font-display font-bold text-[#1A1F36] text-base mb-2">
                  {reason.title}
                </h3>
                <p className="text-gray-600 text-sm leading-relaxed">
                  {reason.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
