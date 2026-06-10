import { motion, useInView } from "framer-motion";
import { useRef } from "react";
import { Quote } from "lucide-react";

const ESPERANZA_IMG = "/manus-storage/esperanza-cabrera_691680f4.png";

export default function Leadership() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-80px" });

  return (
    <section id="leadership" className="py-24 lg:py-32 bg-white">
      <div className="container" ref={ref}>
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
          className="max-w-2xl mb-16"
        >
          <span className="text-[#E8722A] font-semibold text-sm uppercase tracking-wider mb-3 block">
            Leadership
          </span>
          <h2 className="font-display text-3xl sm:text-4xl lg:text-5xl font-bold text-[#1A1F36] mb-5">
            Led by Excellence
          </h2>
          <p className="text-gray-600 text-lg leading-relaxed">
            Our leadership team brings deep expertise in outbound sales, operations management,
            and client success across diverse industries.
          </p>
        </motion.div>

        {/* Leader Card */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.7, delay: 0.2 }}
          className="grid grid-cols-1 lg:grid-cols-5 gap-12 items-center"
        >
          {/* Image */}
          <div className="lg:col-span-2">
            <div className="relative">
              <img
                src={ESPERANZA_IMG}
                alt="Esperanza Cabrera, Founder & CEO of Swift Dialers"
                className="rounded-2xl shadow-2xl shadow-[#1A1F36]/15 w-full max-w-sm mx-auto lg:mx-0 object-cover aspect-[3/4]"
              />
              {/* Decorative elements */}
              <div className="absolute -bottom-4 -right-4 w-full h-full rounded-2xl border-2 border-[#E8722A]/20 -z-10" />
            </div>
          </div>

          {/* Content */}
          <div className="lg:col-span-3">
            <div className="relative">
              <Quote size={48} className="text-[#E8722A]/20 mb-4" />
              <blockquote className="text-xl lg:text-2xl text-[#1A1F36] font-display font-medium leading-relaxed mb-8 italic">
                "We don't just make calls — we open doors. Every conversation is an opportunity
                to create value for our clients, and our team treats each dial with the same
                dedication and professionalism as if it were their own business on the line."
              </blockquote>
            </div>

            <div className="border-t border-gray-100 pt-6">
              <h3 className="font-display text-2xl font-bold text-[#1A1F36] mb-1">
                Esperanza Cabrera
              </h3>
              <p className="text-[#E8722A] font-semibold mb-4">
                Founder & CEO
              </p>
              <p className="text-gray-600 leading-relaxed max-w-xl">
                With years of experience in outbound sales and BPO operations, Esperanza built
                Swift Dialers from the ground up with a singular vision: to create an elite team
                that delivers results others cannot. Under her leadership, the team has expanded
                across multiple industries and consistently exceeded client expectations.
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
