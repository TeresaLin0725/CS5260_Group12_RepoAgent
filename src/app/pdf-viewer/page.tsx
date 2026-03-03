'use client';

import React, { useEffect, useState, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import ThemeToggle from '@/components/theme-toggle';
import LanguageToggle from '@/components/LanguageToggle';
import FloatingChat from '@/components/FloatingChat';
import { useLanguage } from '@/contexts/LanguageContext';

function PdfViewerContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const { messages } = useLanguage();
  const pdfViewer = messages.pdfViewer;

  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [filename, setFilename] = useState<string>('report.pdf');
  const [error, setError] = useState<string | null>(null);

  // Repo info passed via query params for the AI assistant
  const repoUrl = searchParams.get('repoUrl') || '';
  const repoType = searchParams.get('repoType') || 'github';
  const repoToken = searchParams.get('token') || '';
  const repoProvider = searchParams.get('provider') || '';
  const repoModel = searchParams.get('model') || '';
  const repoLanguage = searchParams.get('language') || 'en';
  const repoName = searchParams.get('repoName') || '';

  useEffect(() => {
    // Retrieve the PDF blob from sessionStorage (set before navigation)
    const pdfBase64 = sessionStorage.getItem('pdfViewerData');
    const storedFilename = sessionStorage.getItem('pdfViewerFilename');

    if (pdfBase64) {
      try {
        // Convert base64 back to blob
        const byteCharacters = atob(pdfBase64);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: 'application/pdf' });
        const url = URL.createObjectURL(blob);
        setPdfUrl(url);
        if (storedFilename) setFilename(storedFilename);
      } catch (err) {
        console.error('Error loading PDF:', err);
        setError(pdfViewer?.loadError || 'Failed to load PDF. Please try generating it again.');
      }
    } else {
      setError(pdfViewer?.noPdf || 'No PDF found. Please generate a PDF report first.');
    }

    return () => {
      if (pdfUrl) URL.revokeObjectURL(pdfUrl);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleDownload = () => {
    if (!pdfUrl) return;
    const a = document.createElement('a');
    a.href = pdfUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="min-h-screen bg-[var(--background)] text-[var(--foreground)] flex flex-col">
      {/* Top bar */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-[var(--border-color)] bg-[var(--card-bg)]">
        <div className="flex items-center gap-4">
          <button
            onClick={() => router.push('/')}
            className="text-sm text-[var(--foreground)]/70 hover:text-[var(--foreground)] transition-colors flex items-center gap-1.5"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            {pdfViewer?.backToHome || 'Back to Home'}
          </button>
          <div className="h-5 w-px bg-[var(--border-color)]" />
          <h1 className="text-sm font-semibold truncate max-w-[400px]">
            {repoName || filename}
          </h1>
        </div>
        <div className="flex items-center gap-3">
          {pdfUrl && (
            <button
              onClick={handleDownload}
              className="text-sm px-3 py-1.5 rounded-lg bg-gradient-to-r from-teal-600 to-cyan-500 text-white hover:shadow-md transition-all flex items-center gap-1.5"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              {pdfViewer?.download || 'Download PDF'}
            </button>
          )}
          <LanguageToggle />
          <ThemeToggle />
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1 flex items-center justify-center p-4">
        {error ? (
          <div className="text-center">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-red-100 dark:bg-red-900/30 mb-4">
              <svg className="w-8 h-8 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
            </div>
            <p className="text-gray-600 dark:text-gray-400 mb-4">{error}</p>
            <button
              onClick={() => router.push('/')}
              className="px-4 py-2 rounded-lg bg-gradient-to-r from-teal-600 to-cyan-500 text-white hover:shadow-md transition-all"
            >
              {pdfViewer?.goGenerate || 'Go Generate PDF'}
            </button>
          </div>
        ) : pdfUrl ? (
          <iframe
            src={pdfUrl}
            className="w-full h-[calc(100vh-5rem)] rounded-lg border border-[var(--border-color)]"
            title="PDF Report"
          />
        ) : (
          <div className="flex items-center gap-2 text-gray-500">
            <div className="w-5 h-5 border-2 border-teal-500 border-t-transparent rounded-full animate-spin" />
            {pdfViewer?.loading || 'Loading PDF...'}
          </div>
        )}
      </main>

      {/* Floating AI Chat Assistant */}
      {repoUrl && (
        <FloatingChat
          repoUrl={repoUrl}
          repoType={repoType}
          token={repoToken || undefined}
          provider={repoProvider}
          model={repoModel}
          language={repoLanguage}
        />
      )}
    </div>
  );
}

export default function PdfViewerPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen flex items-center justify-center bg-[var(--background)]">
        <div className="flex items-center gap-2 text-gray-500">
          <div className="w-5 h-5 border-2 border-teal-500 border-t-transparent rounded-full animate-spin" />
          Loading...
        </div>
      </div>
    }>
      <PdfViewerContent />
    </Suspense>
  );
}
