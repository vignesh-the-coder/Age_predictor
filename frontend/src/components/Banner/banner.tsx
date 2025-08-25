import { Button } from "@heroui/react";

export default function Banner() {
  return (
    <section className="w-full bg-gradient-to-r from-blue-600 to-indigo-700 text-white rounded-2xl shadow-lg overflow-hidden">
      <div className="mx-auto max-w-7xl px-6 py-6 lg:flex lg:items-center lg:justify-between lg:gap-12">
        
        {/* Left Content */}
        <div className="flex-1">
          <h1 className="text-4xl font-bold sm:text-5xl lg:text-6xl">
            Predict Age with AI 🚀
          </h1>
          <p className="mt-6 text-lg text-blue-100 max-w-xl">
            Upload a photo and let our AI model predict age instantly. 
            Built with React, TailwindCSS, and HeroUI.
          </p>

          <div className="mt-8 flex flex-wrap gap-4">
            <Button
              as="a"
              href="#get-started"
              size="lg"
              color="primary"
              radius="lg"
              className="font-semibold"
            >
              Get Started
            </Button>
            <Button
              as="a"
              href="#learn-more"
              size="lg"
              variant="bordered"
              radius="lg"
              className="text-white border-white font-semibold"
            >
              Learn More
            </Button>
          </div>
        </div>

        {/* Right Image */}
        <div className="mt-12 lg:mt-0 lg:flex-1 lg:flex lg:justify-end">
          <img
            src="age2.jpg"
            alt="AI illustration"
            className="w-full max-w-md rounded-xl shadow-xl hidden sm:block"
          />
        </div>
      </div>
    </section>
  );
}
