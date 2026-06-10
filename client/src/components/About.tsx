import { motion, useInView } from "framer-motion";
import { useRef } from "react";
import { CheckCircle2 } from "lucide-react";

const TEAM_IMG = "https://d2xsxph8kpxj0f.cloudfront.net/310419663029858924/mrBS9fE8HNLUBDvLdivFBd/team-working-EKTvxdBecUR9DrtvQjkP72.webp";
const NOMAD_IMG = "https://d2xsxph8kpxj0f.cloudfront.net/310419663029858924/mrBS9fE8HNLUBDvLdivFBd/digital-nomad-ZxiStFWcf9Dm27io9JfDcy.webp";

const highlights = [
  "Elite team of trained outbound specialists",
  "Digital nomad workforce — flexible and always-on",
  "Proven results in restricted advertising industries",
  "Dedicated account management for every client",
  "Scalable operations from 1 to 50+ agents",
  "Full transparency with real-time reporting",
];

export default function About() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-80px" });

  return (
    <section id="about" className="py-24 lg:py-32 bg-white">
      <div className="container">
        <div ref={ref} className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          {/* Left: Images */}
          <motion.div
            initial={{ opacity: 0, x: -40 }}
            animate={isInView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.7 }}
            className="relative"
          >
            <div className="relative">
              <img
                src={TEAM_IMG}
                alt="Swift Dialers team collaborating"
                className="rounded-2xl shadow-2xl shadow-[#1A1F36]/10 w-full object-cover aspect-[4/3]"
              />
              {/* Floating second image */}
              <div className="absolute -bottom-8 -right-4 lg:-right-8 w-48 lg:w-56">
                <img
                  src={NOMAD_IMG}
                  alt="Digital nomad professional"
                  className="rounded-xl shadow-xl border-4 border-white object-cover aspect-square"
                />
              </div>
              {/* Decorative accent */}
              <div className="absolute -top-4 -left-4 w-24 h-24 bg-[#E8722A]/10 rounded-2xl -z-10" />
            </div>
          </motion.div>

          {/* Right: Content */}
          <motion.div
            initial={{ opacity: 0, x: 40 }}
            animate={isInView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.7, delay: 0.15 }}
          >
            <span className="text-[#E8722A] font-semibold text-sm uppercase tracking-wider mb-3 block">
              About Swift Dialers
            </span>
            <h2 className="font-display text-3xl sm:text-4xl font-bold text-[#1A1F36] mb-6 leading-tight">
              A Global Team Built for
              <br />
              <span className="text-[#E8722A]">Exceptional Results</span>
            </h2>
            <p className="text-gray-600 text-base leading-relaxed mb-6">
              Swift Dialers is not your typical call center. We are an elite, handpicked team of
              outbound specialists and BPO professionals who operate as digital nomads across the globe.
              Every member of our team is passionate about delivering measurable outcomes for our clients.
            </p>
            <p className="text-gray-600 text-base leading-relaxed mb-8">
              Led by Esperanza Cabrera, we specialize in industries where traditional online advertising
              is difficult or restricted. Our approach combines relentless work ethic with sophisticated
              outreach strategies to open doors that others cannot.
            </p>

            {/* Highlights */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {highlights.map((item, i) => (
                <motion.div
                  key={item}
                  initial={{ opacity: 0, y: 10 }}
                  animate={isInView ? { opacity: 1, y: 0 } : {}}
                  transition={{ duration: 0.4, delay: 0.3 + i * 0.06 }}
                  className="flex items-start gap-2"
                >
                  <CheckCircle2 size={18} className="text-[#E8722A] mt-0.5 shrink-0" />
                  <span className="text-sm text-[#1A1F36] font-medium">{item}</span>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
