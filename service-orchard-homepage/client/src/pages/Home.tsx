import Navigation from "@/components/Navigation";
import HeroSection from "@/components/HeroSection";
import RolePickerSection from "@/components/RolePickerSection";
import WhatWeDoSection from "@/components/WhatWeDoSection";
import AutomatableSection from "@/components/AutomatableSection";
import SkillPacksSection from "@/components/SkillPacksSection";
import ScanSection from "@/components/ScanSection";
import BridgeSection from "@/components/BridgeSection";
import OfferLadderSection from "@/components/OfferLadderSection";
import HowItWorksSection from "@/components/HowItWorksSection";
import FinalCTASection from "@/components/FinalCTASection";
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navigation />
      <main className="flex-1">
        <HeroSection />
        <RolePickerSection />
        <WhatWeDoSection />
        <AutomatableSection />
        <SkillPacksSection />
        <HowItWorksSection />
        <ScanSection />
        <BridgeSection />
        <OfferLadderSection />
        <FinalCTASection />
      </main>
      <Footer />
    </div>
  );
}
