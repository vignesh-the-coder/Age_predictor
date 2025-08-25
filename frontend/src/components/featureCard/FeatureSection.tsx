



import { Icon } from "@iconify/react";
import React from "react";

type FeatureCategory = {
  key: string;
  title: string;
  icon: React.ReactNode;
  descriptions: string[];
  bgColor: string;
};

const featuresCategories: FeatureCategory[] = [
  {
    key: "examples",
    title: "Examples",
    icon: <Icon icon="solar:mask-happly-linear" width={40} />,
    descriptions: [
        "Upload a selfie to get an instant age prediction",
      "Try different photos to see how age estimation changes",
      "Fun and interactive demo of AI in action",
    ],
    bgColor: "bg-sky-100",
  },
  {
    key: "capabilities",
    title: "Capabilities",
    icon: <Icon icon="solar:magic-stick-3-linear" width={40} />,
    descriptions: [
          "Provides instant age estimation from images",
      "Works on desktop, tablet, and mobile devices",
      "No images are stored — your privacy is safe",

    ],
    bgColor: "bg-purple-100",
  },
  {
    key: "limitations",
    title: "Limitations",
    icon: <Icon icon="solar:shield-warning-outline" width={40} />,
    descriptions: [
       "Predictions are approximate and may not always be accurate",
      "Image quality, lighting, and expressions affect results",
      "Built for fun and learning, not as a professional tool",

    ],
    // Tailwind doesn’t include beige, so use inline style
    bgColor: "bg-[beige]",
  },
];

function FeatureCard({
  title,
  icon,
  descriptions,
  bgColor,
}: {
  title: string;
  icon: React.ReactNode;
  descriptions: string[];
  bgColor: string;
}) {
  return (
    <div className={`p-6 rounded-2xl shadow-md ${bgColor}`}>
     
      <div className="flex items-center gap-3 mb-4">
        {icon}
        <h3 className="text-xl font-semibold">{title}</h3>
      </div>
      <ul className="space-y-2 text-gray-700">
        {descriptions.map((desc, index) => (
          <li key={index} className="text-sm">
            • {desc}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function FeatureSection() {
  return (
     <section
      id="about"   // 👈 add this
      className="py-16 px-6"
    >
    <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 md:grid-cols-3">
      {featuresCategories.map((category) => (
        <FeatureCard
          key={category.key}
          descriptions={category.descriptions}
          icon={category.icon}
          title={category.title}
          bgColor={category.bgColor}
        />
      ))}
    </div>
    </section>
  );
}
