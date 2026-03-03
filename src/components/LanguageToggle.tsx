"use client";

import { useLanguage } from "@/contexts/LanguageContext";

export default function LanguageToggle() {
  const { language, setLanguage } = useLanguage();

  return (
    <button
      type="button"
      className="cursor-pointer bg-transparent border border-[var(--border-color)] text-[var(--foreground)] hover:border-[var(--accent-primary)] active:bg-[var(--accent-secondary)]/10 rounded-md px-2.5 py-2 transition-all duration-300 text-xs font-medium tracking-wide"
      title={language === "en" ? "切换到中文" : "Switch to English"}
      aria-label={language === "en" ? "Switch to Chinese" : "Switch to English"}
      onClick={() => setLanguage(language === "en" ? "zh" : "en")}
    >
      {language === "en" ? "中文" : "EN"}
    </button>
  );
}
