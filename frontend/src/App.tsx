
// import Navbar from "./components/Navbar/Navbar";
// import Banner from "./components/Banner/banner";
// import Hero from "./components/Hero/hero";
// import FeatureSection from "./components/featureCard/FeatureSection";
//  // renamed Component → FeatureSection

// export default function App() {
//   return (
//     <div className="flex flex-col w-full">
//       {/* Navbar */}
//       <Navbar />

//       {/* Banner */}
//       <Banner />

//       {/* Hero Section */}
//       <Hero />

//       {/* Feature Cards Section */}
//       <section className="w-full px-6 py-12 bg-white">
//         <FeatureSection />
//       </section>

//     </div>
//   );
// }







"use client";



// Existing components
import Navbar from "./components/Navbar/Navbar";
import Banner from "./components/Banner/banner";
import Hero from "./components/Hero/hero";
import FeatureSection from "./components/featureCard/FeatureSection";
import Faq from "./components/Faq/Faq";


import ImagePredict from "./components/image-predict/AgePredict";


export default function App() {


  return (
    <div className="flex flex-col w-full">
      {/* Navbar */}
      <Navbar />

      {/* Banner */}
      <Banner />

      {/* Hero Section */}
      <Hero />

      {/* Feature Cards Section */}
      <section className="w-full px-6 py-12 bg-white">
        <FeatureSection />
      </section>

      {/* Prediction + Feedback Section */}

           <ImagePredict />
           
            <Faq />

    </div>
  );
}
