'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import ThemeToggle from '@/components/theme-toggle';
import LanguageToggle from '@/components/LanguageToggle';
import Mermaid from '../components/Mermaid';
import ConfigurationModal from '@/components/ConfigurationModal';
import { extractUrlPath, extractUrlDomain } from '@/utils/urlDecoder';

import { useLanguage } from '@/contexts/LanguageContext';

// Define the demo mermaid charts outside the component
const DEMO_FLOW_CHART = `graph TD
  A[Code Repository] --> B[RepoHelper]
  B --> C[Architecture Diagrams]
  B --> D[Component Relationships]
  B --> E[Data Flow]
  B --> F[Process Workflows]

  style A fill:#d4e8ec,stroke:#4a8c99
  style B fill:#b8d8e0,stroke:#3d7a8a
  style C fill:#d0e8d4,stroke:#4a9960
  style D fill:#cce0f0,stroke:#4a7c99
  style E fill:#e0d4d8,stroke:#996070
  style F fill:#d8e8cc,stroke:#6a9940`;

const DEMO_SEQUENCE_CHART = `sequenceDiagram
  participant User
  participant RepoHelper
  participant GitHub

  User->>RepoHelper: Enter repository URL
  RepoHelper->>GitHub: Request repository data
  GitHub-->>RepoHelper: Return repository data
  RepoHelper->>RepoHelper: Process and analyze code
  RepoHelper-->>User: Display results with diagrams

  %% Add a note to make text more visible
  Note over User,GitHub: RepoHelper supports sequence diagrams for visualizing interactions`;

export default function Home() {
  const router = useRouter();
  const { language, setLanguage, messages, supportedLanguages } = useLanguage();

  // Create a simple translation function
  const t = (key: string, params: Record<string, string | number> = {}): string => {
    // Split the key by dots to access nested properties
    const keys = key.split('.');
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let value: any = messages;

    // Navigate through the nested properties
    for (const k of keys) {
      if (value && typeof value === 'object' && k in value) {
        value = value[k];
      } else {
        // Return the key if the translation is not found
        return key;
      }
    }

    // If the value is a string, replace parameters
    if (typeof value === 'string') {
      return Object.entries(params).reduce((acc: string, [paramKey, paramValue]) => {
        return acc.replace(`{${paramKey}}`, String(paramValue));
      }, value);
    }

    // Return the key if the value is not a string
    return key;
  };

  const [repositoryInput, setRepositoryInput] = useState('');

  const REPO_CONFIG_CACHE_KEY = 'repohelperRepoConfigCache';

  const loadConfigFromCache = (repoUrl: string) => {
    if (!repoUrl) return;
    try {
      const cachedConfigs = localStorage.getItem(REPO_CONFIG_CACHE_KEY);
      if (cachedConfigs) {
        const configs = JSON.parse(cachedConfigs);
        const config = configs[repoUrl.trim()];
        if (config) {
          setSelectedLanguage(config.selectedLanguage || language);
          setProvider(config.provider || '');
          setModel(config.model || '');
          setIsCustomModel(config.isCustomModel || false);
          setCustomModel(config.customModel || '');
          setSelectedPlatform(config.selectedPlatform || 'github');
          setExcludedDirs(config.excludedDirs || '');
          setExcludedFiles(config.excludedFiles || '');
          setIncludedDirs(config.includedDirs || '');
          setIncludedFiles(config.includedFiles || '');
        }
      }
    } catch (error) {
      console.error('Error loading config from localStorage:', error);
    }
  };

  const handleRepositoryInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newRepoUrl = e.target.value;
    setRepositoryInput(newRepoUrl);
    if (newRepoUrl.trim() === "") {
      // Optionally reset fields if input is cleared
    } else {
        loadConfigFromCache(newRepoUrl);
    }
  };

  useEffect(() => {
    if (repositoryInput) {
      loadConfigFromCache(repositoryInput);
    }
  }, []);

  // Provider-based model selection state
  const [provider, setProvider] = useState<string>('');
  const [model, setModel] = useState<string>('');
  const [isCustomModel, setIsCustomModel] = useState<boolean>(false);
  const [customModel, setCustomModel] = useState<string>('');

  // Export configuration state

  const [excludedDirs, setExcludedDirs] = useState('');
  const [excludedFiles, setExcludedFiles] = useState('');
  const [includedDirs, setIncludedDirs] = useState('');
  const [includedFiles, setIncludedFiles] = useState('');
  const [selectedPlatform, setSelectedPlatform] = useState<'github' | 'gitlab' | 'bitbucket'>('github');
  const [accessToken, setAccessToken] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [selectedLanguage, setSelectedLanguage] = useState<string>(language);

  // Authentication state
  const [authRequired, setAuthRequired] = useState<boolean>(false);
  const [authCode, setAuthCode] = useState<string>('');
  const [isAuthLoading, setIsAuthLoading] = useState<boolean>(true);

  // Direct PDF generation state
  const [isPdfGenerating, setIsPdfGenerating] = useState(false);
  const [pdfPhase, setPdfPhase] = useState<string | null>(null);
  const [pdfError, setPdfError] = useState<string | null>(null);

  // Direct PPT generation state
  const [isPptGenerating, setIsPptGenerating] = useState(false);
  const [pptPhase, setPptPhase] = useState<string | null>(null);
  const [pptError, setPptError] = useState<string | null>(null);

  // Direct Video generation state
  const [isVideoGenerating, setIsVideoGenerating] = useState(false);
  const [videoPhase, setVideoPhase] = useState<string | null>(null);
  const [videoError, setVideoError] = useState<string | null>(null);

  // Sync the language context with the selectedLanguage state
  useEffect(() => {
    setLanguage(selectedLanguage);
  }, [selectedLanguage]);

  // Reverse sync: when language toggle changes context language, update local selectedLanguage
  useEffect(() => {
    setSelectedLanguage(language);
  }, [language]);

  // Fetch authentication status on component mount
  useEffect(() => {
    const fetchAuthStatus = async () => {
      try {
        setIsAuthLoading(true);
        const response = await fetch('/api/auth/status');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setAuthRequired(data.auth_required);
      } catch (err) {
        console.error("Failed to fetch auth status:", err);
        // Assuming auth is required if fetch fails to avoid blocking UI for safety
        setAuthRequired(true);
      } finally {
        setIsAuthLoading(false);
      }
    };

    fetchAuthStatus();
  }, []);

  // Parse repository URL/input and extract owner and repo
  const parseRepositoryInput = (input: string): {
    owner: string,
    repo: string,
    type: string,
    fullPath?: string,
    localPath?: string
  } | null => {
    input = input.trim();

    let owner = '', repo = '', type = 'github', fullPath;
    let localPath: string | undefined;

    // Handle Windows absolute paths (e.g., C:\path\to\folder)
    const windowsPathRegex = /^[a-zA-Z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*$/;
    const customGitRegex = /^(?:https?:\/\/)?([^\/]+)\/(.+?)\/([^\/]+)(?:\.git)?\/?$/;

    if (windowsPathRegex.test(input)) {
      type = 'local';
      localPath = input;
      repo = input.split('\\').pop() || 'local-repo';
      owner = 'local';
    }
    // Handle Unix/Linux absolute paths (e.g., /path/to/folder)
    else if (input.startsWith('/')) {
      type = 'local';
      localPath = input;
      repo = input.split('/').filter(Boolean).pop() || 'local-repo';
      owner = 'local';
    }
    else if (customGitRegex.test(input)) {
      // Detect repository type based on domain
      const domain = extractUrlDomain(input);
      if (domain?.includes('github.com')) {
        type = 'github';
      } else if (domain?.includes('gitlab.com') || domain?.includes('gitlab.')) {
        type = 'gitlab';
      } else if (domain?.includes('bitbucket.org') || domain?.includes('bitbucket.')) {
        type = 'bitbucket';
      } else {
        type = 'web'; // fallback for other git hosting services
      }

      fullPath = extractUrlPath(input)?.replace(/\.git$/, '');
      const parts = fullPath?.split('/') ?? [];
      if (parts.length >= 2) {
        repo = parts[parts.length - 1] || '';
        owner = parts[parts.length - 2] || '';
      }
    }
    // Unsupported URL formats
    else {
      console.error('Unsupported URL format:', input);
      return null;
    }

    if (!owner || !repo) {
      return null;
    }

    // Clean values
    owner = owner.trim();
    repo = repo.trim();

    // Remove .git suffix if present
    if (repo.endsWith('.git')) {
      repo = repo.slice(0, -4);
    }

    return { owner, repo, type, fullPath, localPath };
  };

  // State for configuration modal
  const [isConfigModalOpen, setIsConfigModalOpen] = useState(false);

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Parse repository input to validate
    const parsedRepo = parseRepositoryInput(repositoryInput);

    if (!parsedRepo) {
      setError('Invalid repository format. Use "owner/repo", GitHub/GitLab/BitBucket URL, or a local folder path like "/path/to/folder" or "C:\\path\\to\\folder".');
      return;
    }

    // If valid, open the configuration modal
    setError(null);
    setIsConfigModalOpen(true);
  };

  const validateAuthCode = async () => {
    try {
      if(authRequired) {
        if(!authCode) {
          return false;
        }
        const response = await fetch('/api/auth/validate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({'code': authCode})
        });
        if (!response.ok) {
          return false;
        }
        const data = await response.json();
        return data.success || false;
      }
    } catch {
      return false;
    }
    return true;
  };

  // Direct PDF generation handler
  const handleGeneratePdf = async () => {
    const parsedRepo = parseRepositoryInput(repositoryInput);
    if (!parsedRepo) {
      setPdfError('Invalid repository format.');
      return;
    }

    // Check authorization if needed
    const validation = await validateAuthCode();
    if (!validation) {
      setPdfError('Failed to validate authorization code.');
      return;
    }

    setIsPdfGenerating(true);
    setPdfError(null);
    setPdfPhase('Phase 1/3: Preparing embeddings...');

    try {
      const { type, localPath } = parsedRepo;
      const repoUrl = localPath || repositoryInput.trim();
      const repoName = `${parsedRepo.owner}/${parsedRepo.repo}`;
      const repoType = type === 'local' ? 'local' : selectedPlatform;

      setPdfPhase('Phase 2/3: Generating report...');

      const response = await fetch('/api/export/repo-pdf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          repo_url: repoUrl,
          repo_name: repoName,
          provider: provider || 'openai',
          model: (isCustomModel && customModel) ? customModel : (model || null),
          language: selectedLanguage,
          repo_type: repoType,
          access_token: accessToken || null,
          excluded_dirs: excludedDirs || null,
          excluded_files: excludedFiles || null,
          included_dirs: includedDirs || null,
          included_files: includedFiles || null,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text().catch(() => 'No error details');
        throw new Error(`PDF generation failed: ${response.status} - ${errorText}`);
      }

      setPdfPhase('Phase 3/3: Downloading PDF...');

      const contentDisposition = response.headers.get('Content-Disposition');
      let filename = `${parsedRepo.repo}_report.pdf`;
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename=(.+)/);
        if (filenameMatch && filenameMatch[1]) {
          filename = filenameMatch[1].replace(/"/g, '');
        }
      }

      const blob = await response.blob();

      // 1. Auto-download the PDF
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      // 2. Store PDF in sessionStorage and navigate to viewer page with AI assistant
      const blobReader = new FileReader();
      blobReader.onloadend = () => {
        const base64 = (blobReader.result as string).split(',')[1];
        sessionStorage.setItem('pdfViewerData', base64);
        sessionStorage.setItem('pdfViewerFilename', filename);

        const params = new URLSearchParams({
          repoUrl: repoUrl,
          repoType: repoType,
          repoName: repoName,
          provider: provider || 'openai',
          model: (isCustomModel && customModel) ? customModel : (model || ''),
          language: selectedLanguage,
          ...(accessToken ? { token: accessToken } : {}),
        });
        router.push(`/pdf-viewer?${params.toString()}`);
      };
      blobReader.readAsDataURL(blob);
    } catch (err) {
      console.error('Error generating PDF:', err);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error during PDF generation';
      setPdfError(errorMessage);
    } finally {
      setIsPdfGenerating(false);
      setPdfPhase(null);
    }
  };

  // Direct PPT generation handler
  const handleGeneratePpt = async () => {
    const parsedRepo = parseRepositoryInput(repositoryInput);
    if (!parsedRepo) {
      setPptError('Invalid repository format.');
      return;
    }

    const validation = await validateAuthCode();
    if (!validation) {
      setPptError('Failed to validate authorization code.');
      return;
    }

    setIsPptGenerating(true);
    setPptError(null);
    setPptPhase('Phase 1/3: Preparing embeddings...');

    try {
      const { type, localPath } = parsedRepo;
      const repoUrl = localPath || repositoryInput.trim();
      const repoName = `${parsedRepo.owner}/${parsedRepo.repo}`;
      const repoType = type === 'local' ? 'local' : selectedPlatform;

      setPptPhase('Phase 2/3: Generating presentation...');

      const response = await fetch('/api/export/repo-ppt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          repo_url: repoUrl,
          repo_name: repoName,
          provider: provider || 'openai',
          model: (isCustomModel && customModel) ? customModel : (model || null),
          language: selectedLanguage,
          repo_type: repoType,
          access_token: accessToken || null,
          excluded_dirs: excludedDirs || null,
          excluded_files: excludedFiles || null,
          included_dirs: includedDirs || null,
          included_files: includedFiles || null,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text().catch(() => 'No error details');
        throw new Error(`PPT generation failed: ${response.status} - ${errorText}`);
      }

      setPptPhase('Phase 3/3: Downloading PPT...');

      const contentDisposition = response.headers.get('Content-Disposition');
      let filename = `${parsedRepo.repo}_slides.pptx`;
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename=(.+)/);
        if (filenameMatch && filenameMatch[1]) {
          filename = filenameMatch[1].replace(/"/g, '');
        }
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error('Error generating PPT:', err);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error during PPT generation';
      setPptError(errorMessage);
    } finally {
      setIsPptGenerating(false);
      setPptPhase(null);
    }
  };

  // Direct Video generation handler
  const handleGenerateVideo = async () => {
    const parsedRepo = parseRepositoryInput(repositoryInput);
    if (!parsedRepo) {
      setVideoError('Invalid repository format.');
      return;
    }

    const validation = await validateAuthCode();
    if (!validation) {
      setVideoError('Failed to validate authorization code.');
      return;
    }

    setIsVideoGenerating(true);
    setVideoError(null);
    setVideoPhase('Phase 1/3: Preparing embeddings...');

    try {
      const { type, localPath } = parsedRepo;
      const repoUrl = localPath || repositoryInput.trim();
      const repoName = `${parsedRepo.owner}/${parsedRepo.repo}`;
      const repoType = type === 'local' ? 'local' : selectedPlatform;

      setVideoPhase('Phase 2/3: Generating video...');

      const response = await fetch('/api/export/repo-video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          repo_url: repoUrl,
          repo_name: repoName,
          provider: provider || 'openai',
          model: (isCustomModel && customModel) ? customModel : (model || null),
          language: selectedLanguage,
          repo_type: repoType,
          access_token: accessToken || null,
          excluded_dirs: excludedDirs || null,
          excluded_files: excludedFiles || null,
          included_dirs: includedDirs || null,
          included_files: includedFiles || null,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text().catch(() => 'No error details');
        throw new Error(`Video generation failed: ${response.status} - ${errorText}`);
      }

      setVideoPhase('Phase 3/3: Downloading Video...');

      const contentDisposition = response.headers.get('Content-Disposition');
      let filename = `${parsedRepo.repo}_overview.mp4`;
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename=(.+)/);
        if (filenameMatch && filenameMatch[1]) {
          filename = filenameMatch[1].replace(/"/g, '');
        }
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error('Error generating Video:', err);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error during Video generation';
      setVideoError(errorMessage);
    } finally {
      setIsVideoGenerating(false);
      setVideoPhase(null);
    }
  };

  return (
    <div className="min-h-screen bg-[var(--background)] flex flex-col">
      {/* Top navigation bar */}
      <nav className="sticky top-0 z-30 backdrop-blur-md bg-[var(--background)]/80 border-b border-[var(--border-color)]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[var(--accent-primary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <span className="text-base font-semibold text-[var(--foreground)] tracking-tight">{t('common.appName')}</span>
            <span className="hidden sm:inline text-xs text-[var(--muted)] ml-1">{t('common.tagline')}</span>
          </div>
          <div className="flex items-center gap-2">
            <LanguageToggle />
            <ThemeToggle />
          </div>
        </div>
      </nav>

      {/* Hero section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-[var(--accent-primary)]/[0.04] to-transparent pointer-events-none" />
        <div className="max-w-3xl mx-auto px-4 sm:px-6 pt-16 pb-12 text-center relative">
          <h1 className="text-3xl sm:text-4xl font-bold text-[var(--foreground)] tracking-tight mb-3">
            {t('home.welcome')}
          </h1>
          <p className="text-[var(--muted)] text-base sm:text-lg mb-8 max-w-xl mx-auto leading-relaxed">
            {t('home.description')}
          </p>

          {/* Search-style input */}
          <form onSubmit={handleFormSubmit} className="max-w-2xl mx-auto">
            <div className="relative group">
              <div className="absolute inset-0 bg-[var(--accent-primary)]/10 rounded-xl blur-xl opacity-0 group-focus-within:opacity-100 transition-opacity duration-500" />
              <div className="relative flex items-center bg-[var(--card-bg)] rounded-xl border border-[var(--border-color)] shadow-custom group-focus-within:border-[var(--accent-primary)] transition-all">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[var(--muted)] ml-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                <input
                  type="text"
                  value={repositoryInput}
                  onChange={handleRepositoryInputChange}
                  placeholder={t('form.repoPlaceholder') || "owner/repo, GitHub/GitLab/BitBucket URL, or local folder path"}
                  className="flex-1 bg-transparent px-3 py-3.5 text-[var(--foreground)] placeholder-[var(--muted)]/60 focus:outline-none text-sm"
                />
                <button
                  type="submit"
                  className="btn-japanese mr-1.5 px-5 py-2 rounded-lg text-sm disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0"
                >
                  {t('common.configure') || 'Configure'}
                </button>
              </div>
            </div>
            {error && (
              <div className="text-[var(--highlight)] text-xs mt-2 text-left pl-2">
                {error}
              </div>
            )}
          </form>
        </div>
      </section>

      {/* Configuration Modal */}
      <ConfigurationModal
        isOpen={isConfigModalOpen}
        onClose={() => setIsConfigModalOpen(false)}
        repositoryInput={repositoryInput}
        selectedLanguage={selectedLanguage}
        setSelectedLanguage={setSelectedLanguage}
        supportedLanguages={supportedLanguages}
        provider={provider}
        setProvider={setProvider}
        model={model}
        setModel={setModel}
        isCustomModel={isCustomModel}
        setIsCustomModel={setIsCustomModel}
        customModel={customModel}
        setCustomModel={setCustomModel}
        selectedPlatform={selectedPlatform}
        setSelectedPlatform={setSelectedPlatform}
        accessToken={accessToken}
        setAccessToken={setAccessToken}
        excludedDirs={excludedDirs}
        setExcludedDirs={setExcludedDirs}
        excludedFiles={excludedFiles}
        setExcludedFiles={setExcludedFiles}
        includedDirs={includedDirs}
        setIncludedDirs={setIncludedDirs}
        includedFiles={includedFiles}
        setIncludedFiles={setIncludedFiles}
        authRequired={authRequired}
        authCode={authCode}
        setAuthCode={setAuthCode}
        isAuthLoading={isAuthLoading}
        onGeneratePdf={handleGeneratePdf}
        isPdfGenerating={isPdfGenerating}
        pdfPhase={pdfPhase}
        pdfError={pdfError}
        onGeneratePpt={handleGeneratePpt}
        isPptGenerating={isPptGenerating}
        pptPhase={pptPhase}
        pptError={pptError}
        onGenerateVideo={handleGenerateVideo}
        isVideoGenerating={isVideoGenerating}
        videoPhase={videoPhase}
        videoError={videoError}
      />

      {/* Main content */}
      <main className="flex-1">
        {/* Feature introduction module */}
        <section className="max-w-5xl mx-auto px-4 sm:px-6 py-10">
          <div className="flex items-center gap-2 mb-6">
            <div className="h-px flex-1 bg-[var(--border-color)]" />
            <span className="text-xs font-medium text-[var(--muted)] uppercase tracking-wider">{t('home.featuresTitle')}</span>
            <div className="h-px flex-1 bg-[var(--border-color)]" />
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              {
                icon: "M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z",
                label: t('home.featureMultiModel'),
                desc: t('home.featureMultiModelDesc'),
              },
              {
                icon: "M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z",
                label: t('home.featureCustomRepo'),
                desc: t('home.featureCustomRepoDesc'),
              },
              {
                icon: "M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z",
                label: t('home.featureAiChat'),
                desc: t('home.featureAiChatDesc'),
              },
              {
                icon: "M12 10v6m0 0l-3-3m3 3l3-3M3 17V7a2 2 0 012-2h6l2 2h6a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z",
                label: t('home.featureExport'),
                desc: t('home.featureExportDesc'),
                badges: ['PDF', 'PPT', 'Video'],
              },
            ].map((item, i) => (
              <div key={i} className="group bg-[var(--card-bg)] rounded-lg border border-[var(--border-color)] p-4 hover:border-[var(--accent-primary)]/40 hover:shadow-custom transition-all duration-300">
                <div className="w-9 h-9 rounded-lg bg-[var(--accent-primary)]/10 flex items-center justify-center mb-3 group-hover:bg-[var(--accent-primary)]/15 transition-colors">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4.5 w-4.5 text-[var(--accent-primary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d={item.icon} />
                  </svg>
                </div>
                <h3 className="text-sm font-medium text-[var(--foreground)] mb-1">{item.label}</h3>
                <p className="text-xs text-[var(--muted)] leading-relaxed">{item.desc}</p>
                {'badges' in item && item.badges && (
                  <div className="flex flex-wrap gap-1.5 mt-2">
                    {item.badges.map((badge, j) => (
                      <span key={j} className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-[var(--accent-primary)]/10 text-[var(--accent-primary)]">
                        {badge}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>

        {/* Supported formats */}
        <section className="max-w-5xl mx-auto px-4 sm:px-6 pb-8">
          <h3 className="text-xs font-medium text-[var(--muted)] uppercase tracking-wider mb-3">
            {t('home.enterRepoUrl')}
          </h3>
          <div className="flex flex-wrap gap-2">
            {[
              "https://github.com/owner/repo",
              "https://gitlab.com/group/project",
              "https://bitbucket.org/team/repo",
              "owner/repo",
            ].map((example, i) => (
              <button
                key={i}
                type="button"
                onClick={() => setRepositoryInput(example)}
                className="text-xs font-mono px-3 py-1.5 rounded-md bg-[var(--card-bg)] border border-[var(--border-color)] text-[var(--muted)] hover:text-[var(--foreground)] hover:border-[var(--accent-primary)]/40 transition-all cursor-pointer"
              >
                {example}
              </button>
            ))}
          </div>
        </section>

        {/* Diagram showcase — side by side on desktop */}
        <section className="max-w-5xl mx-auto px-4 sm:px-6 pb-12">
          <div className="flex items-center gap-2 mb-4">
            <div className="h-px flex-1 bg-[var(--border-color)]" />
            <span className="text-xs font-medium text-[var(--muted)] uppercase tracking-wider">{t('home.advancedVisualization')}</span>
            <div className="h-px flex-1 bg-[var(--border-color)]" />
          </div>
          <p className="text-sm text-[var(--muted)] text-center mb-6 max-w-xl mx-auto">
            {t('home.diagramDescription')}
          </p>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            <div className="bg-[var(--card-bg)] rounded-lg border border-[var(--border-color)] overflow-hidden">
              <div className="px-4 py-2.5 border-b border-[var(--border-color)] bg-[var(--background)]/50">
                <h4 className="text-xs font-medium text-[var(--muted)] uppercase tracking-wider">{t('home.flowDiagram')}</h4>
              </div>
              <div className="p-4">
                <Mermaid chart={DEMO_FLOW_CHART} />
              </div>
            </div>
            <div className="bg-[var(--card-bg)] rounded-lg border border-[var(--border-color)] overflow-hidden">
              <div className="px-4 py-2.5 border-b border-[var(--border-color)] bg-[var(--background)]/50">
                <h4 className="text-xs font-medium text-[var(--muted)] uppercase tracking-wider">{t('home.sequenceDiagram')}</h4>
              </div>
              <div className="p-4">
                <Mermaid chart={DEMO_SEQUENCE_CHART} />
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-[var(--border-color)] bg-[var(--card-bg)]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-12 flex items-center justify-between">
          <p className="text-[var(--muted)] text-xs">{t('footer.copyright')}</p>
        </div>
      </footer>
    </div>
  );
}