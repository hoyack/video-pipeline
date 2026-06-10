import { motion, useInView } from "framer-motion";
import { useRef } from "react";
import {
  Shield,
  Zap,
  ShoppingBag,
  Building2,
  Leaf,
  Heart,
  Home,
  Briefcase,
  Cpu,
  Scale,
} from "lucide-react";

const SERVICES_BG = "https://d2xsxph8kpxj0f.cloudfront.net/310419663029858924/mrBS9fE8HNLUBDvLdivFBd/services-bg-ZaDdtUFm2v4S95EFFpmG9o.webp";

const industries = [
  { icon: Shield, name: "Insurance", description: "Life, health, and property insurance appointment setting" },
  { icon: Zap, name: "Energy & Solar", description: "Renewable energy and utility service outreach" },
  { icon: Home, name: "Home Services", description: "HVAC, roofing, plumbing, and home improvement" },
  { icon: Heart, name: "Healthcare", description: "Medical practice outreach and patient engagement" },
  { icon: Scale, name: "Legal Services", description: "Law firm lead generation and intake support" },
  { icon: Leaf, name: "CBD & Wellness", description: "Restricted-advertising industry specialists" },
  { icon: Cpu, name: "Technology", description: "SaaS, IT services, and tech product outreach" },
  { icon: ShoppingBag, name: "Retail & E-Commerce", description: "Customer support and order management" },
  { icon: Building2, name: "Real Estate", description: "Property listing outreach and lead qualification" },
  { icon: Briefcase, name: "Financial Services", description: "Wealth management and fintech outbound" },
];

export default function Industries() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-80px" });

  return (
    <section
      id="industries"
      className="relative py-24 lg:py-32 overflow-hidden"
    >
      {/* Background */}
      <div className="absolute inset-0">
        <img src={SERVICES_BG} alt="" className="w-full h-full object-cover" />
        <div className="absolute inset-0 bg-[#1A1F36]/85" />
      </div>

      <div className="container relative z-10" ref={ref}>
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
          className="text-center max-w-2xl mx-auto mb-16"
        >
          <span className="text-[#E8722A] font-semibold text-sm uppercase tracking-wider mb-3 block">
            Industries We Serve
          </span>
          <h2 className="font-display text-3xl sm:text-4xl lg:text-5xl font-bold text-white mb-5">
            Expertise Across Sectors
          </h2>
          <p className="text-white/70 text-lg leading-relaxed">
            We specialize in industries where traditional online advertising is restricted or
            ineffective — and we excel in mainstream sectors too.
          </p>
        </motion.div>

        {/* Industries Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
          {industries.map((industry, index) => (
            <motion.div
              key={industry.name}
              initial={{ opacity: 0, y: 20 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.4, delay: 0.1 + index * 0.05 }}
              className="group relative p-6 rounded-xl bg-white/5 backdrop-blur-sm border border-white/10 hover:bg-white/10 hover:border-[#E8722A]/40 transition-all duration-300 text-center"
            >
              <industry.icon
                size={28}
                className="mx-auto mb-3 text-[#E8722A] group-hover:scale-110 transition-transform duration-300"
              />
              <h3 className="text-white font-semibold text-sm mb-1">
                {industry.name}
              </h3>
              <p className="text-white/50 text-xs leading-relaxed hidden sm:block">
                {industry.description}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
