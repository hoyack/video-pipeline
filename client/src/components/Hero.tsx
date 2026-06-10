import { motion } from "framer-motion";
import { ArrowRight, Phone, Globe, TrendingUp } from "lucide-react";

const HERO_BG = "https://d2xsxph8kpxj0f.cloudfront.net/310419663029858924/mrBS9fE8HNLUBDvLdivFBd/hero-bg-AmsjkgqP9hnv2aDjTq4WR3.webp";

export default function Hero() {
  return (
    <section className="relative min-h-screen flex items-center overflow-hidden">
      {/* Background Image with Overlay */}
      <div className="absolute inset-0">
        <img
          src={HERO_BG}
          alt=""
          className="w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-r from-[#1A1F36]/90 via-[#1A1F36]/75 to-[#1A1F36]/40" />
        <div className="absolute inset-0 bg-gradient-to-t from-[#1A1F36]/60 via-transparent to-transparent" />
      </div>

      {/* Content */}
      <div className="container relative z-10 pt-28 pb-20 lg:pt-32 lg:pb-24">
        <div className="max-w-3xl">
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/10 backdrop-blur-sm border border-white/20 mb-8"
          >
            <span className="w-2 h-2 rounded-full bg-[#E8722A] animate-pulse" />
            <span className="text-white/90 text-sm font-medium">Elite BPO & Appointment Setting</span>
          </motion.div>

          {/* Headline */}
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.3 }}
            className="font-display text-4xl sm:text-5xl lg:text-6xl xl:text-7xl font-bold text-white leading-[1.08] mb-6"
          >
            Your Dedicated
            <br />
            <span className="text-[#E8722A]">Outbound Team,</span>
            <br />
            Anywhere in the World
          </motion.h1>

          {/* Subheadline */}
          <motion.p
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.45 }}
            className="text-lg sm:text-xl text-white/80 leading-relaxed max-w-2xl mb-10"
          >
            Swift Dialers is a small elite group of outbound appointment setters and BPO specialists.
            We operate as digital nomads, passionately delivering outstanding results for clients across industries where others cannot reach.
          </motion.p>

          {/* CTAs */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.6 }}
            className="flex flex-col sm:flex-row gap-4"
          >
            <a
              href="#contact"
              className="inline-flex items-center justify-center gap-2 px-8 py-4 bg-[#E8722A] text-white font-semibold rounded-full hover:bg-[#d4631f] transition-all duration-200 hover:shadow-xl hover:shadow-[#E8722A]/25 active:scale-[0.97] text-base"
            >
              Schedule a Consultation
              <ArrowRight size={18} />
            </a>
            <a
              href="#services"
              className="inline-flex items-center justify-center gap-2 px-8 py-4 bg-white/10 backdrop-blur-sm text-white font-semibold rounded-full border border-white/25 hover:bg-white/20 transition-all duration-200 text-base"
            >
              Explore Our Services
            </a>
          </motion.div>

          {/* Stats Row */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.8 }}
            className="mt-16 grid grid-cols-3 gap-6 max-w-lg"
          >
            <div className="flex flex-col items-start">
              <div className="flex items-center gap-2 mb-1">
                <Phone size={16} className="text-[#E8722A]" />
                <span className="text-2xl sm:text-3xl font-bold text-white font-display">50K+</span>
              </div>
              <span className="text-white/60 text-xs sm:text-sm">Calls Monthly</span>
            </div>
            <div className="flex flex-col items-start">
              <div className="flex items-center gap-2 mb-1">
                <Globe size={16} className="text-[#E8722A]" />
                <span className="text-2xl sm:text-3xl font-bold text-white font-display">12+</span>
              </div>
              <span className="text-white/60 text-xs sm:text-sm">Industries Served</span>
            </div>
            <div className="flex flex-col items-start">
              <div className="flex items-center gap-2 mb-1">
                <TrendingUp size={16} className="text-[#E8722A]" />
                <span className="text-2xl sm:text-3xl font-bold text-white font-display">98%</span>
              </div>
              <span className="text-white/60 text-xs sm:text-sm">Client Retention</span>
            </div>
          </motion.div>
        </div>
      </div>

      {/* Bottom gradient fade */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-[#F8FAFB] to-transparent" />
    </section>
  );
}
