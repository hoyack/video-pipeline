import { motion, useInView } from "framer-motion";
import { useRef, useState, useEffect } from "react";

const stats = [
  { value: 500000, suffix: "+", label: "Calls Completed", prefix: "" },
  { value: 98, suffix: "%", label: "Client Retention Rate", prefix: "" },
  { value: 12, suffix: "+", label: "Industries Served", prefix: "" },
  { value: 35, suffix: "%", label: "Avg. Conversion Lift", prefix: "" },
];

function AnimatedCounter({
  value,
  suffix,
  prefix,
  isInView,
}: {
  value: number;
  suffix: string;
  prefix: string;
  isInView: boolean;
}) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    if (!isInView) return;
    const duration = 2000;
    const steps = 60;
    const increment = value / steps;
    let current = 0;
    const timer = setInterval(() => {
      current += increment;
      if (current >= value) {
        setCount(value);
        clearInterval(timer);
      } else {
        setCount(Math.floor(current));
      }
    }, duration / steps);
    return () => clearInterval(timer);
  }, [isInView, value]);

  const formatted = count >= 1000 ? `${Math.floor(count / 1000)}K` : count.toString();

  return (
    <span className="font-display text-4xl sm:text-5xl lg:text-6xl font-bold text-white">
      {prefix}
      {formatted}
      {suffix}
    </span>
  );
}

export default function Results() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-80px" });

  return (
    <section className="py-20 lg:py-24 bg-[#1A1F36]">
      <div className="container" ref={ref}>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <span className="text-[#E8722A] font-semibold text-sm uppercase tracking-wider mb-3 block">
            Proven Results
          </span>
          <h2 className="font-display text-3xl sm:text-4xl font-bold text-white">
            Numbers That Speak for Themselves
          </h2>
        </motion.div>

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-8 lg:gap-12">
          {stats.map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 30 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="text-center"
            >
              <AnimatedCounter
                value={stat.value}
                suffix={stat.suffix}
                prefix={stat.prefix}
                isInView={isInView}
              />
              <p className="text-white/60 text-sm mt-2 font-medium">
                {stat.label}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
