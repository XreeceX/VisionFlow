import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        background: "#050816",
        foreground: "#E5E7EB",
        accent: {
          DEFAULT: "#7C3AED",
          soft: "#4C1D95"
        }
      },
      boxShadow: {
        "glow": "0 0 40px rgba(124,58,237,0.45)"
      },
      backgroundImage: {
        "radial-grid":
          "radial-gradient(circle at 1px 1px, rgba(148,163,184,0.15) 1px, transparent 0)"
      },
      animation: {
        "pulse-slow": "pulse 3s ease-in-out infinite",
        "spin-slow": "spin 6s linear infinite"
      }
    }
  },
  plugins: []
};

export default config;

