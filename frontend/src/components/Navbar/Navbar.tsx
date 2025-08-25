
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import clsx from "clsx";

const links = [
  { name: "Home", href: "#" },
  { name: "About", href: "#about" },
  { name: "Services", href: "#services" },
 
];

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <nav className="fixed top-0 left-0 w-full bg-white shadow-md z-50">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 md:px-6">
        {/* Logo */}
        <div className="text-2xl font-bold text-blue-600">Age Predictor</div>

        {/* Desktop Menu */}
        <div className="hidden md:flex space-x-6">
          {links.map((link) => (
            <a
              key={link.name}
              href={link.href}
              className="text-gray-700 hover:text-blue-600 transition-colors"
            >
              {link.name}
            </a>
          ))}
        </div>

        {/* Mobile Hamburger */}
        <button
          className="md:hidden flex flex-col space-y-1"
          onClick={() => setIsOpen((prev) => !prev)}
        >
          <span
            className={clsx(
              "block h-0.5 w-6 bg-gray-700 transition-all",
              isOpen && "rotate-45 translate-y-1.5"
            )}
          />
          <span
            className={clsx(
              "block h-0.5 w-6 bg-gray-700 transition-all",
              isOpen && "opacity-0"
            )}
          />
          <span
            className={clsx(
              "block h-0.5 w-6 bg-gray-700 transition-all",
              isOpen && "-rotate-45 -translate-y-1.5"
            )}
          />
        </button>
      </div>

      {/* Mobile Dropdown */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0 }}
            animate={{ height: "auto" }}
            exit={{ height: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden md:hidden bg-gray-50 shadow-inner"
          >
            <div className="flex flex-col space-y-4 px-4 py-4">
              {links.map((link) => (
                <a
                  key={link.name}
                  href={link.href}
                  className="text-gray-700 hover:text-blue-600 transition-colors"
                  onClick={() => setIsOpen(false)}
                >
                  {link.name}
                </a>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
}
