import { motion, useInView } from "framer-motion";
import { useRef, useState } from "react";
import { Send, Mail, MapPin, Clock } from "lucide-react";
import { toast } from "sonner";

export default function Contact() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-80px" });
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    company: "",
    message: "",
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    toast.success("Thank you! We'll be in touch within 24 hours.");
    setFormData({ name: "", email: "", company: "", message: "" });
  };

  return (
    <section id="contact" className="py-24 lg:py-32 bg-[#F8FAFB]">
      <div className="container" ref={ref}>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16">
          {/* Left: Info */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={isInView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.6 }}
          >
            <span className="text-[#E8722A] font-semibold text-sm uppercase tracking-wider mb-3 block">
              Get In Touch
            </span>
            <h2 className="font-display text-3xl sm:text-4xl lg:text-5xl font-bold text-[#1A1F36] mb-6">
              Ready to Scale Your
              <br />
              Outbound Operations?
            </h2>
            <p className="text-gray-600 text-lg leading-relaxed mb-10">
              Whether you need appointment setters, customer support, or back-office specialists,
              we're ready to deploy a dedicated team tailored to your needs.
            </p>

            {/* Contact Info */}
            <div className="space-y-6">
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 rounded-xl bg-[#E8722A]/10 flex items-center justify-center shrink-0">
                  <Mail size={18} className="text-[#E8722A]" />
                </div>
                <div>
                  <p className="font-semibold text-[#1A1F36] text-sm">Email Us</p>
                  <p className="text-gray-600 text-sm">hello@swiftdialers.com</p>
                </div>
              </div>
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 rounded-xl bg-[#E8722A]/10 flex items-center justify-center shrink-0">
                  <MapPin size={18} className="text-[#E8722A]" />
                </div>
                <div>
                  <p className="font-semibold text-[#1A1F36] text-sm">Location</p>
                  <p className="text-gray-600 text-sm">Global — Digital Nomad Team</p>
                </div>
              </div>
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 rounded-xl bg-[#E8722A]/10 flex items-center justify-center shrink-0">
                  <Clock size={18} className="text-[#E8722A]" />
                </div>
                <div>
                  <p className="font-semibold text-[#1A1F36] text-sm">Response Time</p>
                  <p className="text-gray-600 text-sm">Within 24 hours, guaranteed</p>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Right: Form */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={isInView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.6, delay: 0.15 }}
          >
            <form
              onSubmit={handleSubmit}
              className="bg-white rounded-2xl p-8 lg:p-10 shadow-xl shadow-[#1A1F36]/5 border border-gray-100"
            >
              <div className="space-y-5">
                <div>
                  <label className="block text-sm font-semibold text-[#1A1F36] mb-2">
                    Full Name
                  </label>
                  <input
                    type="text"
                    required
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    className="w-full px-4 py-3 rounded-xl border border-gray-200 bg-[#F8FAFB] text-[#1A1F36] placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-[#E8722A]/30 focus:border-[#E8722A] transition-all text-sm"
                    placeholder="Your name"
                  />
                </div>
                <div>
                  <label className="block text-sm font-semibold text-[#1A1F36] mb-2">
                    Email Address
                  </label>
                  <input
                    type="email"
                    required
                    value={formData.email}
                    onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                    className="w-full px-4 py-3 rounded-xl border border-gray-200 bg-[#F8FAFB] text-[#1A1F36] placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-[#E8722A]/30 focus:border-[#E8722A] transition-all text-sm"
                    placeholder="you@company.com"
                  />
                </div>
                <div>
                  <label className="block text-sm font-semibold text-[#1A1F36] mb-2">
                    Company
                  </label>
                  <input
                    type="text"
                    value={formData.company}
                    onChange={(e) => setFormData({ ...formData, company: e.target.value })}
                    className="w-full px-4 py-3 rounded-xl border border-gray-200 bg-[#F8FAFB] text-[#1A1F36] placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-[#E8722A]/30 focus:border-[#E8722A] transition-all text-sm"
                    placeholder="Your company name"
                  />
                </div>
                <div>
                  <label className="block text-sm font-semibold text-[#1A1F36] mb-2">
                    How Can We Help?
                  </label>
                  <textarea
                    required
                    rows={4}
                    value={formData.message}
                    onChange={(e) => setFormData({ ...formData, message: e.target.value })}
                    className="w-full px-4 py-3 rounded-xl border border-gray-200 bg-[#F8FAFB] text-[#1A1F36] placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-[#E8722A]/30 focus:border-[#E8722A] transition-all text-sm resize-none"
                    placeholder="Tell us about your needs — number of agents, industry, timeline..."
                  />
                </div>
                <button
                  type="submit"
                  className="w-full flex items-center justify-center gap-2 px-8 py-4 bg-[#E8722A] text-white font-semibold rounded-xl hover:bg-[#d4631f] transition-all duration-200 hover:shadow-lg hover:shadow-[#E8722A]/20 active:scale-[0.98] text-sm"
                >
                  <Send size={16} />
                  Send Message
                </button>
              </div>
            </form>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
