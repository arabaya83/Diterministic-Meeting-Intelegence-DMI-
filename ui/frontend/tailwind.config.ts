import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#0F172A",
        surface: "#111827",
        card: "#1F2937",
        border: "#334155",
        textPrimary: "#F1F5F9",
        textSecondary: "#94A3B8",
        accent: "#3B82F6",
        success: "#10B981",
        warning: "#F59E0B",
        error: "#EF4444",
      },
      boxShadow: {
        panel: "0 12px 40px rgba(15, 23, 42, 0.28)",
      },
    },
  },
  plugins: [],
} satisfies Config;
