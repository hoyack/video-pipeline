const LOGO_URL = "https://d2xsxph8kpxj0f.cloudfront.net/310419663029858924/mrBS9fE8HNLUBDvLdivFBd/swift-dialers-logo-5Vnszmfy8EaVTWbqb3VXdx.webp";

export default function Footer() {
  return (
    <footer className="bg-[#1A1F36] pt-16 pb-8">
      <div className="container">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-10 mb-12">
          {/* Brand */}
          <div className="md:col-span-2">
            <img
              src={LOGO_URL}
              alt="Swift Dialers"
              className="h-10 w-auto mb-4 brightness-0 invert"
            />
            <p className="text-white/60 text-sm leading-relaxed max-w-sm">
              Swift Dialers is an elite team of outbound appointment setters and BPO specialists
              delivering outstanding results for clients across industries worldwide.
            </p>
          </div>

          {/* Services */}
          <div>
            <h4 className="text-white font-semibold text-sm mb-4">Services</h4>
            <ul className="space-y-2.5">
              <li>
                <a href="#services" className="text-white/50 text-sm hover:text-[#E8722A] transition-colors">
                  Appointment Setting
                </a>
              </li>
              <li>
                <a href="#services" className="text-white/50 text-sm hover:text-[#E8722A] transition-colors">
                  Customer Support
                </a>
              </li>
              <li>
                <a href="#services" className="text-white/50 text-sm hover:text-[#E8722A] transition-colors">
                  Back-Office Operations
                </a>
              </li>
              <li>
                <a href="#services" className="text-white/50 text-sm hover:text-[#E8722A] transition-colors">
                  Data Management
                </a>
              </li>
            </ul>
          </div>

          {/* Company */}
          <div>
            <h4 className="text-white font-semibold text-sm mb-4">Company</h4>
            <ul className="space-y-2.5">
              <li>
                <a href="#about" className="text-white/50 text-sm hover:text-[#E8722A] transition-colors">
                  About Us
                </a>
              </li>
              <li>
                <a href="#leadership" className="text-white/50 text-sm hover:text-[#E8722A] transition-colors">
                  Leadership
                </a>
              </li>
              <li>
                <a href="#industries" className="text-white/50 text-sm hover:text-[#E8722A] transition-colors">
                  Industries
                </a>
              </li>
              <li>
                <a href="#contact" className="text-white/50 text-sm hover:text-[#E8722A] transition-colors">
                  Contact
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom */}
        <div className="border-t border-white/10 pt-8 flex flex-col sm:flex-row justify-between items-center gap-4">
          <p className="text-white/40 text-xs">
            &copy; {new Date().getFullYear()} Swift Dialers. All rights reserved.
          </p>
          <div className="flex gap-6">
            <a href="#" className="text-white/40 text-xs hover:text-[#E8722A] transition-colors">
              Privacy Policy
            </a>
            <a href="#" className="text-white/40 text-xs hover:text-[#E8722A] transition-colors">
              Terms of Service
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
