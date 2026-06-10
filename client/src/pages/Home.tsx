import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import Services from "@/components/Services";
import About from "@/components/About";
import WhyUs from "@/components/WhyUs";
import Industries from "@/components/Industries";
import Results from "@/components/Results";
import Leadership from "@/components/Leadership";
import Contact from "@/components/Contact";
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main>
        <Hero />
        <Services />
        <WhyUs />
        <About />
        <Results />
        <Industries />
        <Leadership />
        <Contact />
      </main>
      <Footer />
    </div>
  );
}
