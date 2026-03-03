/* eslint-disable @typescript-eslint/no-explicit-any */
'use client';

import React, { createContext, useContext, useState, useEffect, useRef, ReactNode } from 'react';
import { locales } from '@/i18n';

type Messages = Record<string, any>;
type LanguageContextType = {
  language: string;
  setLanguage: (lang: string) => void;
  messages: Messages;
  supportedLanguages: Record<string, string>;
};

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export function LanguageProvider({ children }: { children: ReactNode }) {
  // Initialize with 'en'
  const [language, setLanguageState] = useState<string>('en');
  const [messages, setMessages] = useState<Messages>({});
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [supportedLanguages, setSupportedLanguages] = useState<Record<string, string>>({});
  const [defaultLanguage, setDefaultLanguage] = useState('en');

  // Use refs so the setLanguage callback always sees the latest values
  const supportedLanguagesRef = useRef(supportedLanguages);
  const defaultLanguageRef = useRef(defaultLanguage);
  useEffect(() => { supportedLanguagesRef.current = supportedLanguages; }, [supportedLanguages]);
  useEffect(() => { defaultLanguageRef.current = defaultLanguage; }, [defaultLanguage]);

  useEffect(() => {
    const getSupportedLanguages = async () => {
      try {
        const response = await fetch('/api/lang/config');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setSupportedLanguages(data.supported_languages);
        setDefaultLanguage(data.default);
      } catch (err) {
        console.error("Failed to fetch lang config:", err);
        const defaultSupportedLanguages = {
          "en": "English",
          "zh": "中文"
        };
        setSupportedLanguages(defaultSupportedLanguages);
        setDefaultLanguage("en");
      }
    }
    getSupportedLanguages();
  }, []);

  useEffect(() => {
    if (Object.keys(supportedLanguages).length > 0) {
      const loadLanguage = async () => {
        try {
          // Always default to English; only use stored language if user explicitly chose one
          let storedLanguage = 'en';
          if (typeof window !== 'undefined') {
            const saved = localStorage.getItem('language');
            if (saved && Object.keys(supportedLanguages).includes(saved)) {
              storedLanguage = saved;
            } else {
              // First visit or invalid stored value — default to English
              localStorage.setItem('language', 'en');
            }
          }

          const validLanguage = Object.keys(supportedLanguages).includes(storedLanguage) ? storedLanguage : defaultLanguage;

          // Load messages for the language
          const langMessages = (await import(`../messages/${validLanguage}.json`)).default;

          setLanguageState(validLanguage);
          setMessages(langMessages);

          // Update HTML lang attribute
          if (typeof document !== 'undefined') {
            document.documentElement.lang = validLanguage;
          }
        } catch (error) {
          console.error('Failed to load language:', error);
          const enMessages = (await import('../messages/en.json')).default;
          setMessages(enMessages);
        } finally {
          setIsLoading(false);
        }
      };
      
      loadLanguage();
    }
  }, [supportedLanguages, defaultLanguage]);

  // Update language and load new messages
  const setLanguage = async (lang: string) => {
    try {
      // Use ref to always get latest supportedLanguages
      const currentSupported = supportedLanguagesRef.current;
      const currentDefault = defaultLanguageRef.current;

      // Also accept any locale from i18n.ts as a fallback check
      const isValid = Object.keys(currentSupported).includes(lang) || locales.includes(lang as any);
      const validLanguage = isValid ? lang : currentDefault;

      console.log('Switching language to:', validLanguage);

      // Load messages for the new language
      const langMessages = (await import(`../messages/${validLanguage}.json`)).default;

      setLanguageState(validLanguage);
      setMessages(langMessages);

      // Store in localStorage
      if (typeof window !== 'undefined') {
        localStorage.setItem('language', validLanguage);
      }

      // Update HTML lang attribute
      if (typeof document !== 'undefined') {
        document.documentElement.lang = validLanguage;
      }
    } catch (error) {
      console.error('Failed to set language:', error);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-100 dark:bg-gray-900">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-teal-500 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <LanguageContext.Provider value={{ language, setLanguage, messages, supportedLanguages }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  const context = useContext(LanguageContext);
  if (context === undefined) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
}
