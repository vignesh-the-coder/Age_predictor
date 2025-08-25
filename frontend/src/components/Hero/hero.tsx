
import { Button } from "@heroui/react";
import { motion } from "framer-motion";

export default function Hero() {
  return (
    <section className="w-full min-h-[70vh] flex flex-col justify-center items-center text-center px-6 bg-gradient-to-b from-[#F4F4F5] via-white to-[#F5A524]/10">
      {/* Title */}
      <motion.h1
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-4xl md:text-6xl font-extrabold leading-tight"
      >
        Predict Age with <span className="text-[#F31260]">AI</span>
      </motion.h1>

      {/* Subtitle */}
      <motion.p
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="mt-4 text-lg md:text-xl text-gray-600 max-w-2xl"
      >
        Unlock insights using <span className="text-[#17C964]">machine learning</span>  
        & modern data tools.
      </motion.p>

      {/* CTA Buttons */}
      <div className="mt-6 flex gap-4 flex-wrap justify-center">
        <Button
          style={{ backgroundColor: "#F31260", color: "white" }}
          size="lg"
          radius="full"
          className="text-lg px-8 py-4 rounded-full"
        >
          Get Started
        </Button>
        <Button
          style={{ borderColor: "#17C964", color: "#17C964" }}
          variant="bordered"
          size="lg"
          radius="full"
          className="text-lg px-8 py-4 rounded-full"
        >
          Learn More
        </Button>
      </div>
    </section>
  );
}
