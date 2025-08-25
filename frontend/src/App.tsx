
"use client";



// Existing components
import Navbar from "./components/Navbar/Navbar";
import Banner from "./components/Banner/banner";
import Hero from "./components/Hero/hero";
import FeatureSection from "./components/featureCard/FeatureSection";
import Faq from "./components/Faq/Faq";


import ImagePredict from "./components/image-predict/AgePredict";
import VoicePredict from "./components/voice-predict/voicePredict";


export default function App() {


  return (
    <div className="flex flex-col w-full">
      {/* Navbar */}
      <Navbar />
      <Banner />
      <main className="pt-7">
          <Hero />
      <section className="w-full px-6 py-3 pt-2 mt-3 bg-white">
        <h2 className="text-3xl font-bold text-center mt-3 mb-1 text-[#39387d]">
          How It Works
        </h2>

        <FeatureSection />
      </section>
        <ImagePredict />
        <section className="w-full bg-[#f7437a] mt-5">
        <VoicePredict />
      </section>
        <Faq />
      </main>
   

    </div>
  );
}
