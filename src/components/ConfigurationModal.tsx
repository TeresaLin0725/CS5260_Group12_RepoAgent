'use client';

import React, { useState } from 'react';
import { useLanguage } from '@/contexts/LanguageContext';
import UserSelector from './UserSelector';
import TokenInput from './TokenInput';
import { FaComments } from 'react-icons/fa';

interface ConfigurationModalProps {
  isOpen: boolean;
  onClose: () => void;

  // Repository input
  repositoryInput: string;

  // Language selection
  selectedLanguage: string;
  setSelectedLanguage: (value: string) => void;
  supportedLanguages: Record<string, string>;

  // Model selection
  provider: string;
  setProvider: (value: string) => void;
  model: string;
  setModel: (value: string) => void;
  isCustomModel: boolean;
  setIsCustomModel: (value: boolean) => void;
  customModel: string;
  setCustomModel: (value: string) => void;

  // Platform selection
  selectedPlatform: 'github' | 'gitlab' | 'bitbucket';
  setSelectedPlatform: (value: 'github' | 'gitlab' | 'bitbucket') => void;

  // Access token
  accessToken: string;
  setAccessToken: (value: string) => void;

  // File filter options
  excludedDirs: string;
  setExcludedDirs: (value: string) => void;
  excludedFiles: string;
  setExcludedFiles: (value: string) => void;
  includedDirs: string;
  setIncludedDirs: (value: string) => void;
  includedFiles: string;
  setIncludedFiles: (value: string) => void;

  // Authentication
  authRequired?: boolean;
  authCode?: string;
  setAuthCode?: (code: string) => void;
  isAuthLoading?: boolean;

  // Direct PDF generation
  onGeneratePdf?: () => void;
  isPdfGenerating?: boolean;
  pdfPhase?: string | null;
  pdfError?: string | null;

  // Direct PPT generation
  onGeneratePpt?: () => void;
  isPptGenerating?: boolean;
  pptPhase?: string | null;
  pptError?: string | null;

  // Direct Video generation
  onGenerateVideo?: () => void;
  isVideoGenerating?: boolean;
  videoPhase?: string | null;
  videoError?: string | null;

  // Start chat mode
  onStartChat?: () => void;
}

export default function ConfigurationModal({
  isOpen,
  onClose,
  repositoryInput,
  selectedLanguage,
  setSelectedLanguage,
  supportedLanguages,
  provider,
  setProvider,
  model,
  setModel,
  isCustomModel,
  setIsCustomModel,
  customModel,
  setCustomModel,
  selectedPlatform,
  setSelectedPlatform,
  accessToken,
  setAccessToken,
  excludedDirs,
  setExcludedDirs,
  excludedFiles,
  setExcludedFiles,
  includedDirs,
  setIncludedDirs,
  includedFiles,
  setIncludedFiles,
  authRequired,
  authCode,
  setAuthCode,
  isAuthLoading,
  onStartChat
}: ConfigurationModalProps) {
  const { messages: t } = useLanguage();

  // Show token section state
  const [showTokenSection, setShowTokenSection] = useState(false);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex min-h-screen items-center justify-center p-4 text-center bg-black/50">
        <div className="relative transform overflow-hidden rounded-lg bg-[var(--card-bg)] text-left shadow-xl transition-all sm:my-8 sm:max-w-2xl sm:w-full">
          {/* Modal header with close button */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--border-color)]">
            <h3 className="text-lg font-medium text-[var(--accent-primary)]">
              <span className="text-[var(--accent-primary)]">{t.form?.configureExport || 'Configure Export'}</span>
            </h3>
            <button
              type="button"
              onClick={onClose}
              className="text-[var(--muted)] hover:text-[var(--foreground)] focus:outline-none transition-colors"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Modal body */}
          <div className="p-6 max-h-[70vh] overflow-y-auto">
            {/* Repository info */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-[var(--foreground)] mb-2">
                {t.form?.repository || 'Repository'}
              </label>
              <div className="bg-[var(--background)]/70 p-3 rounded-md border border-[var(--border-color)] text-sm text-[var(--foreground)]">
                {repositoryInput}
              </div>
            </div>

            {/* Language selection */}
            <div className="mb-4">
              <label htmlFor="language-select" className="block text-sm font-medium text-[var(--foreground)] mb-2">
                {t.form?.wikiLanguage || t.form?.language || 'Language'}
              </label>
              <select
                id="language-select"
                value={selectedLanguage}
                onChange={(e) => setSelectedLanguage(e.target.value)}
                className="input-japanese block w-full px-3 py-2 text-sm rounded-md bg-transparent text-[var(--foreground)] focus:outline-none focus:border-[var(--accent-primary)]"
              >
                {
                  Object.entries(supportedLanguages).map(([key, value])=> <option key={key} value={key}>{value}</option>)
                }
              </select>
            </div>

            {/* Model Selector */}
            <div className="mb-4">
              <UserSelector
                provider={provider}
                setProvider={setProvider}
                model={model}
                setModel={setModel}
                isCustomModel={isCustomModel}
                setIsCustomModel={setIsCustomModel}
                customModel={customModel}
                setCustomModel={setCustomModel}
                showFileFilters={true}
                excludedDirs={excludedDirs}
                setExcludedDirs={setExcludedDirs}
                excludedFiles={excludedFiles}
                setExcludedFiles={setExcludedFiles}
                includedDirs={includedDirs}
                setIncludedDirs={setIncludedDirs}
                includedFiles={includedFiles}
                setIncludedFiles={setIncludedFiles}
              />
            </div>

            {/* Access token section using TokenInput component */}
            <TokenInput
              selectedPlatform={selectedPlatform}
              setSelectedPlatform={setSelectedPlatform}
              accessToken={accessToken}
              setAccessToken={setAccessToken}
              showTokenSection={showTokenSection}
              onToggleTokenSection={() => setShowTokenSection(!showTokenSection)}
              allowPlatformChange={true}
            />

            {/* Authorization Code Input */}
            {isAuthLoading && (
              <div className="mb-4 p-3 bg-[var(--background)]/50 rounded-md border border-[var(--border-color)] text-sm text-[var(--muted)]">
                Loading authentication status...
              </div>
            )}
            {!isAuthLoading && authRequired && (
              <div className="mb-4 p-4 bg-[var(--background)]/50 rounded-md border border-[var(--border-color)]">
                <label htmlFor="authCode" className="block text-sm font-medium text-[var(--foreground)] mb-2">
                  {t.form?.authorizationCode || 'Authorization Code'}
                </label>
                <input
                  type="password"
                  id="authCode"
                  value={authCode || ''}
                  onChange={(e) => setAuthCode?.(e.target.value)}
                  className="input-japanese block w-full px-3 py-2 text-sm rounded-md bg-transparent text-[var(--foreground)] focus:outline-none focus:border-[var(--accent-primary)]"
                  placeholder="Enter your authorization code"
                />
                 <div className="flex items-center mt-2 text-xs text-[var(--muted)]">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1 text-[var(--muted)]"
                    fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                   {t.form?.authorizationRequired || 'Authentication is required.'}
                </div>
              </div>
            )}
          </div>

          {/* Modal footer */}
          <div className="flex flex-col gap-2 px-6 py-4 border-t border-[var(--border-color)]">
            <div className="flex items-center justify-end gap-2 flex-wrap">
              {onStartChat && (
                <button
                  type="button"
                  onClick={onStartChat}
                  className="flex items-center gap-1.5 px-5 py-2 text-sm font-medium rounded-md bg-gradient-to-r from-teal-600 to-cyan-500 text-white hover:shadow-md transition-all"
                >
                  <FaComments className="text-xs" />
                  {t.common?.startChat || 'Start Chat'}
                </button>
              )}
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 text-sm font-medium rounded-md border border-[var(--border-color)]/50 text-[var(--muted)] bg-transparent hover:bg-[var(--background)] hover:text-[var(--foreground)] transition-colors"
              >
                {t.common?.cancel || 'Cancel'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
