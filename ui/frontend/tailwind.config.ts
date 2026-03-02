import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#F7F5FA",
        surface: "#FFFFFF",
        card: "#FBFAFD",
        border: "#D8D2E7",
        textPrimary: "#2E2A33",
        textSecondary: "#5F5A66",
        accent: "#4E2A84",
        success: "#008656",
        warning: "#B27A00",
        error: "#B3203D",
      },
      boxShadow: {
        panel: "0 18px 42px rgba(78, 42, 132, 0.10)",
      },
    },
  },
  plugins: [],
} satisfies Config;
