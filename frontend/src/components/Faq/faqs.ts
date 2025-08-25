// faqs.tsx

export interface FAQ {
  title: string;
  content: string;
}

const faqs: FAQ[] = [
  {
    title: "What is this platform about?",
    content:
      "This platform uses AI to analyze images and predict attributes such as age. It’s designed to be simple, interactive, and user-friendly.",
  },
  {
    title: "How accurate are the predictions?",
    content:
      "Predictions are for demonstration purposes only and may not always be precise. The system is trained on sample data and is not a replacement for professional tools.",
  },
  {
    title: "Can I use this on mobile devices?",
    content:
      "Yes! The platform is fully responsive and works across desktops, tablets, and smartphones.",
  },
  {
    title: "Do you store my uploaded images?",
    content:
      "No, uploaded images are processed locally in your browser session. Nothing is stored or shared on external servers.",
  },
  {
    title: "How can I share my feedback?",
    content:
      "After completing a prediction, you’ll be presented with a feedback panel where you can quickly share your thoughts.",
  },
];

export default faqs;
