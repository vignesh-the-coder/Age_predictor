"use client";
import { Accordion, AccordionItem } from "@heroui/react";
import { Icon } from "@iconify/react";
import faqs from "./faqs"; // <-- array of FAQs

export default function Faq() {
  return (
    <section className="w-full  bg-gradient-to-b from-[#F4F4F5] via-white to-[#F5A524]/10 py-20 mt-7 rounded-md  sm:py-32 lg:py-40">
      <div className="mx-auto w-full max-w-6xl px-4 sm:px-6 lg:px-8">
        {/* Centered Heading */}
        <h2 className="text-3xl font-semibold text-center mb-8">     
          <span className="inline-block md:hidden text-[#39387d]">FAQs</span>
          <span className="hidden md:inline-block text-[#39387d]">
            Frequently asked questions
          </span>
        </h2>

        {/* Accordion */}
        <Accordion
          fullWidth
          keepContentMounted
          className="gap-4"
          itemClasses={{
            base: "rounded-xl border border-gray-100 px-6  text-2xl  text-[#39387d] shadow-sm hover:bg-[#ecf1f7] transition-colors",
            title: "font-medium text-lg",
            trigger: "py-5",
            content:
              "pt-0 pb-5 text-base  text-[#39387d] hover:bg-[#ecf1f7] rounded-lg px-4", // <-- open content bg
          }}
          selectionMode="multiple"
          variant="splitted"
        >
          {faqs.map((item, i) => (
            <AccordionItem
              key={i}
              indicator={<Icon icon="lucide:plus" width={24} />}
              title={item.title}
            >
              {item.content}
            </AccordionItem>
          ))}
        </Accordion>
      </div>
    </section>
  );
}
