import { motion } from "framer-motion";
import { useInView } from "framer-motion";
import { useRef } from "react";
import {
  Phone,
  HeadphonesIcon,
  Calculator,
  Database,
  Package,
  BarChart3,
  Truck,
  ClipboardList,
} from "lucide-react";

const services = [
  {
    icon: Phone,
    title: "Outbound Appointment Setting",
    description:
      "Our core expertise. We connect with decision-makers and book qualified appointments that convert, even in industries where traditional ads cannot reach.",
    highlight: true,
  },
  {
    icon: HeadphonesIcon,
    title: "Retail Customer Support",
    description:
      "Dedicated support teams handling inbound inquiries, order management, and customer satisfaction across retail and e-commerce brands.",
  },
  {
    icon: Calculator,
    title: "Accounting & Bookkeeping",
    description:
      "Precision financial support including accounts payable/receivable, reconciliation, and reporting to keep your books clean.",
  },
  {
    icon: Database,
    title: "Data Entry & Management",
    description:
      "Accurate, high-volume data processing with quality controls that ensure your databases remain clean and actionable.",
  },
  {
    icon: Package,
    title: "Price Book Updating",
    description:
      "Systematic maintenance of pricing databases, catalogs, and product information across your sales channels.",
  },
  {
    icon: BarChart3,
    title: "Inventory Cost Analysis",
    description:
      "Deep-dive analysis into inventory costs, margins, and optimization opportunities to improve your bottom line.",
  },
  {
    icon: Truck,
    title: "Logistics Coordination",
    description:
      "End-to-end logistics support including shipment tracking, vendor coordination, and supply chain communication.",
  },
  {
    icon: ClipboardList,
    title: "General Back-Office",
    description:
      "Comprehensive administrative support covering CRM management, document processing, and operational tasks.",
  },
];

function ServiceCard({
  service,
  index,
}: {
  service: (typeof services)[0];
  index: number;
}) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-50px" });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 30 }}
      animate={isInView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.5, delay: index * 0.08 }}
      className={`group relative p-8 rounded-2xl border transition-all duration-300 hover:shadow-xl hover:shadow-[#1A1F36]/5 hover:-translate-y-1 ${
        service.highlight
          ? "bg-[#1A1F36] border-[#1A1F36] text-white"
          : "bg-white border-gray-100 hover:border-[#E8722A]/30"
      }`}
    >
      <div
        className={`inline-flex items-center justify-center w-12 h-12 rounded-xl mb-5 ${
          service.highlight
            ? "bg-[#E8722A]/20 text-[#E8722A]"
            : "bg-[#F0F4F8] text-[#1A1F36] group-hover:bg-[#E8722A]/10 group-hover:text-[#E8722A]"
        } transition-colors duration-300`}
      >
        <service.icon size={24} />
      </div>
      <h3
        className={`font-display font-bold text-lg mb-3 ${
          service.highlight ? "text-white" : "text-[#1A1F36]"
        }`}
      >
        {service.title}
      </h3>
      <p
        className={`text-sm leading-relaxed ${
          service.highlight ? "text-white/75" : "text-gray-600"
        }`}
      >
        {service.description}
      </p>
    </motion.div>
  );
}

export default function Services() {
  const headerRef = useRef(null);
  const headerInView = useInView(headerRef, { once: true, margin: "-50px" });

  return (
    <section id="services" className="py-24 lg:py-32 bg-[#F8FAFB]">
      <div className="container">
        {/* Section Header */}
        <motion.div
          ref={headerRef}
          initial={{ opacity: 0, y: 30 }}
          animate={headerInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
          className="max-w-2xl mb-16"
        >
          <span className="text-[#E8722A] font-semibold text-sm uppercase tracking-wider mb-3 block">
            Our Services
          </span>
          <h2 className="font-display text-3xl sm:text-4xl lg:text-5xl font-bold text-[#1A1F36] mb-5">
            Comprehensive BPO Solutions
          </h2>
          <p className="text-gray-600 text-lg leading-relaxed">
            From outbound calling campaigns to complex back-office operations, we deliver
            enterprise-grade services with the agility and dedication of a boutique team.
          </p>
        </motion.div>

        {/* Services Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {services.map((service, index) => (
            <ServiceCard key={service.title} service={service} index={index} />
          ))}
        </div>
      </div>
    </section>
  );
}
