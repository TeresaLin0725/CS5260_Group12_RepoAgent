'use client';

import React, { useState, useRef, useEffect, useCallback, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import ThemeToggle from '@/components/theme-toggle';
import LanguageToggle from '@/components/LanguageToggle';
import Markdown from '@/components/Markdown';
import ModelSelectionModal from '@/components/ModelSelectionModal';
import { useLanguage } from '@/contexts/LanguageContext';
import { createChatWebSocket, closeWebSocket, ChatCompletionRequest } from '@/utils/websocketClient';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

// Action types the agent can trigger
type AgentAction = 'GENERATE_PDF' | 'GENERATE_PPT' | 'GENERATE_VIDEO';

interface ActionStatus {
  type: AgentAction;
  messageId?: string;
  status: 'pending' | 'running' | 'done' | 'error';
  phase?: string;
  stepIndex?: number;
  totalSteps?: number;
  note?: string;
  error?: string;
}

interface ActionProgressStep {
  afterMs: number;
  message: string;
}

function AgentChatContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const { messages: i18n } = useLanguage();

  // Read config from URL params
  const repoUrl = searchParams.get('repoUrl') || '';
  const repoType = searchParams.get('repoType') || 'github';
  const repoName = searchParams.get('repoName') || '';
  const token = searchParams.get('token') || '';
  const language = searchParams.get('language') || 'en';
  const excludedDirs = searchParams.get('excludedDirs') || '';
  const excludedFiles = searchParams.get('excludedFiles') || '';
  const includedDirs = searchParams.get('includedDirs') || '';
  const includedFiles = searchParams.get('includedFiles') || '';

  // Model state
  const [selectedProvider, setSelectedProvider] = useState(searchParams.get('provider') || '');
  const [selectedModel, setSelectedModel] = useState(searchParams.get('model') || '');
  const [isCustomModel, setIsCustomModel] = useState(searchParams.get('isCustomModel') === 'true');
  const [customModel, setCustomModel] = useState(searchParams.get('customModel') || '');
  const [isModelSelectionModalOpen, setIsModelSelectionModalOpen] = useState(false);

  // Chat state
  const [question, setQuestion] = useState('');
  const [streamingResponse, setStreamingResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationHistory, setConversationHistory] = useState<Message[]>([]);
  const [useDeepResearch, setUseDeepResearch] = useState(false);

  // Action state — tracks ongoing export actions
  const [actionStatuses, setActionStatuses] = useState<ActionStatus[]>([]);

  const inputRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const webSocketRef = useRef<WebSocket | null>(null);
  const actionTimerRefs = useRef<Record<string, number>>({});
  const messageIdRef = useRef(0);

  const t = i18n;

  // Fetch default model config if none supplied
  useEffect(() => {
    const fetchModel = async () => {
      try {
        const response = await fetch('/api/models/config');
        if (!response.ok) return;
        const data = await response.json();
        if (!selectedProvider && !selectedModel) {
          setSelectedProvider(data.defaultProvider);
          const defaultProv = data.providers.find((p: { id: string }) => p.id === data.defaultProvider);
          if (defaultProv && defaultProv.models.length > 0) {
            setSelectedModel(defaultProv.models[0].id);
          }
        }
      } catch (err) {
        console.error('Failed to fetch model configurations:', err);
      }
    };
    fetchModel();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-scroll on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversationHistory, streamingResponse]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Cleanup websocket on unmount
  useEffect(() => {
    return () => { closeWebSocket(webSocketRef.current); };
  }, []);

  useEffect(() => {
    return () => {
      Object.values(actionTimerRefs.current).forEach((timerId) => window.clearInterval(timerId));
      actionTimerRefs.current = {};
    };
  }, []);

  // ── Detect action tags in assistant response ─────────────
  const parseActions = (content: string): AgentAction[] => {
    const actions: AgentAction[] = [];
    const actionRegex = /\[ACTION:(GENERATE_PDF|GENERATE_PPT|GENERATE_VIDEO)\]/g;
    let match;
    while ((match = actionRegex.exec(content)) !== null) {
      actions.push(match[1] as AgentAction);
    }
    return actions;
  };

  // Strip action tags from display text
  const stripActionTags = (content: string): string => {
    return content.replace(/\[ACTION:(GENERATE_PDF|GENERATE_PPT|GENERATE_VIDEO)\]/g, '').trim();
  };

  const nextMessageId = () => `msg-${++messageIdRef.current}`;

  const actionKey = (action: AgentAction, messageId?: string) => `${messageId || 'global'}:${action}`;


  // ── Handle chat submit ───────────────────────────────────
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || isLoading) return;

    const modePrefix = useDeepResearch ? '[AGENT] [DEEP RESEARCH]' : '[AGENT]';
    const actualContent = `${modePrefix} ${question}`;
    const userMessage: Message = { id: nextMessageId(), role: 'user', content: actualContent };
    const newHistory = [...conversationHistory, userMessage];
    setConversationHistory(newHistory);
    setQuestion('');
    setIsLoading(true);
    setStreamingResponse('');

    const requestBody: ChatCompletionRequest = {
      repo_url: repoUrl,
      type: repoType,
      messages: newHistory.map(msg => ({ role: msg.role, content: msg.content })),
      provider: selectedProvider,
      model: isCustomModel ? customModel : selectedModel,
      language: language,
      excluded_dirs: excludedDirs || undefined,
      excluded_files: excludedFiles || undefined,
    };

    if (token) {
      requestBody.token = token;
    }

    closeWebSocket(webSocketRef.current);
    let fullResponse = '';

    webSocketRef.current = createChatWebSocket(
      requestBody,
      (message: string) => {
        fullResponse += message;
        setStreamingResponse(fullResponse);
      },
      (error: Event) => {
        console.error('WebSocket error:', error);
        fullResponse += '\n\nError: Connection failed. Trying HTTP fallback...';
        setStreamingResponse(fullResponse);
        fallbackToHttp(requestBody, newHistory);
      },
      () => {
        if (fullResponse) {
          const assistantMessage: Message = { id: nextMessageId(), role: 'assistant', content: fullResponse };
          setConversationHistory(prev => [...prev, assistantMessage]);
          setStreamingResponse('');
          const actions = parseActions(fullResponse);
          if (actions.length > 0) {
            actions.forEach(action => executeAction(action, assistantMessage.id));
          }
        }
        setIsLoading(false);
      }
    );
  };

  const fallbackToHttp = async (requestBody: ChatCompletionRequest, history: Message[]) => {
    try {
      const apiResponse = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });

      if (!apiResponse.ok) throw new Error(`API error: ${apiResponse.status}`);

      const reader = apiResponse.body?.getReader();
      const decoder = new TextDecoder();
      if (!reader) throw new Error('Failed to get response reader');

      let fullResponse = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        fullResponse += decoder.decode(value, { stream: true });
        setStreamingResponse(fullResponse);
      }

      const assistantMessage: Message = { id: nextMessageId(), role: 'assistant', content: fullResponse };
      setConversationHistory([...history, assistantMessage]);
      setStreamingResponse('');
      const actions = parseActions(fullResponse);
      if (actions.length > 0) {
        actions.forEach(action => executeAction(action, assistantMessage.id));
      }
    } catch (error) {
      console.error('HTTP fallback error:', error);
      setStreamingResponse(prev => prev + '\n\nError: Failed to get response.');
    } finally {
      setIsLoading(false);
    }
  };

  const clearConversation = () => {
    setConversationHistory([]);
    setStreamingResponse('');
    setQuestion('');
    setActionStatuses([]);
  };

  // Strip [AGENT] tag from display
  const getDisplayContent = (msg: Message): string => {
    return msg.content.replace(/^\[AGENT\]\s*/, '');
  };

  // Action label mapping
  const actionLabel = (action: AgentAction): string => {
    const labels: Record<AgentAction, string> = {
      GENERATE_PDF: t?.agentChat?.generatingPdf || 'PDF Report',
      GENERATE_PPT: t?.agentChat?.generatingPpt || 'PPT Slides',
      GENERATE_VIDEO: t?.agentChat?.generatingVideo || 'Video Overview',
    };
    return labels[action];
  };


  const actionProgressPlan = (action: AgentAction): ActionProgressStep[] => {
    if (action === 'GENERATE_VIDEO') {
      return [
        { afterMs: 0, message: 'Analyzing repository structure...' },
        { afterMs: 8000, message: 'Writing narration script...' },
        { afterMs: 20000, message: 'Rendering scenes & generating audio...' },
        { afterMs: 40000, message: 'Composing final MP4...' },
      ];
    }
    if (action === 'GENERATE_PDF') {
      return [
        { afterMs: 0, message: 'Analyzing repository structure...' },
        { afterMs: 4000, message: 'Generating PDF content...' },
        { afterMs: 10000, message: 'Finalizing PDF export...' },
      ];
    }
    return [
      { afterMs: 0, message: 'Analyzing repository structure...' },
      { afterMs: 5000, message: 'Building slide content...' },
      { afterMs: 12000, message: 'Finalizing presentation export...' },
    ];
  };

  const startActionProgress = useCallback((action: AgentAction, messageId?: string, jobId?: string) => {
    const key = actionKey(action, messageId);
    if (actionTimerRefs.current[key]) {
      window.clearInterval(actionTimerRefs.current[key]);
      delete actionTimerRefs.current[key];
    }

    const note = action === 'GENERATE_VIDEO'
      ? 'Please keep this page open while the video is being generated.'
      : 'Expected wait: usually under a minute.';

    setActionStatuses(prev => [
      ...prev.filter(a => !(a.type === action && a.messageId === messageId)),
      {
        type: action,
        messageId,
        status: 'running',
        phase: 'Starting...',
        stepIndex: 1,
        totalSteps: 5,
        note,
      }
    ]);

    if (action === 'GENERATE_VIDEO' && jobId) {
      // Poll real progress from backend
      const timerId = window.setInterval(async () => {
        try {
          const res = await fetch(`/api/export/progress/${jobId}`);
          if (!res.ok) return;
          const data = await res.json();
          setActionStatuses(prev => prev.map((item) => (
            item.type === action && item.messageId === messageId && item.status === 'running'
              ? { ...item, phase: data.message, stepIndex: data.step, totalSteps: data.total, note }
              : item
          )));
        } catch { /* ignore polling errors */ }
      }, 1500);
      actionTimerRefs.current[key] = timerId;
    } else {
      // Fallback: time-based estimation for PDF/PPT
      const steps = actionProgressPlan(action);
      const startedAt = Date.now();
      setActionStatuses(prev => prev.map((item) => (
        item.type === action && item.messageId === messageId
          ? { ...item, phase: steps[0]?.message || 'Working...', stepIndex: 1, totalSteps: steps.length }
          : item
      )));
      const timerId = window.setInterval(() => {
        const elapsed = Date.now() - startedAt;
        let activeIndex = 0;
        for (let i = 0; i < steps.length; i += 1) {
          if (elapsed >= steps[i].afterMs) activeIndex = i;
        }
        setActionStatuses(prev => prev.map((item) => (
          item.type === action && item.messageId === messageId && item.status === 'running'
            ? { ...item, phase: steps[activeIndex]?.message || item.phase, stepIndex: activeIndex + 1, totalSteps: steps.length, note }
            : item
        )));
      }, 1000);
      actionTimerRefs.current[key] = timerId;
    }
  }, []);

  const stopActionProgress = useCallback((action: AgentAction, messageId?: string) => {
    const key = actionKey(action, messageId);
    const timerId = actionTimerRefs.current[key];
    if (timerId) {
      window.clearInterval(timerId);
      delete actionTimerRefs.current[key];
    }
  }, []);

  // ?? Execute an export action ?????????????????????????????
  const executeAction = useCallback(async (action: AgentAction, messageId?: string) => {
    const exportModel = (isCustomModel && customModel) ? customModel : (selectedModel || null);

    const bodyPayload = {
      repo_url: repoUrl,
      repo_name: repoName,
      provider: selectedProvider || 'openai',
      model: exportModel,
      language: language,
      repo_type: repoType,
      access_token: token || null,
      excluded_dirs: excludedDirs || null,
      excluded_files: excludedFiles || null,
      included_dirs: includedDirs || null,
      included_files: includedFiles || null,
    };

    let endpoint = '';
    let defaultFilename = '';
    if (action === 'GENERATE_PDF') {
      endpoint = '/api/export/repo-pdf';
      defaultFilename = `${repoName.split('/').pop() || 'repo'}_report.pdf`;
    } else if (action === 'GENERATE_PPT') {
      endpoint = '/api/export/repo-ppt';
      defaultFilename = `${repoName.split('/').pop() || 'repo'}_slides.pptx`;
    } else if (action === 'GENERATE_VIDEO') {
      endpoint = '/api/export/repo-video';
      defaultFilename = `${repoName.split('/').pop() || 'repo'}_overview.mp4`;
    }

    // Generate a unique job ID for video progress tracking
    const jobId = action === 'GENERATE_VIDEO' ? `vid_${Date.now()}_${Math.random().toString(36).slice(2, 8)}` : undefined;
    startActionProgress(action, messageId, jobId);

    try {
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      if (jobId) headers['X-Job-Id'] = jobId;

      const response = await fetch(endpoint, {
        method: 'POST',
        headers,
        body: JSON.stringify(bodyPayload),
      });

      if (!response.ok) {
        const errorText = await response.text().catch(() => 'Unknown error');
        throw new Error(`Export failed: ${response.status} - ${errorText}`);
      }

      const contentDisposition = response.headers.get('Content-Disposition');
      let filename = defaultFilename;
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

      stopActionProgress(action, messageId);
      setActionStatuses(prev => prev.map(a => a.type === action && a.messageId === messageId ? { ...a, status: 'done', phase: undefined, stepIndex: undefined, totalSteps: undefined, note: undefined } : a));
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      stopActionProgress(action, messageId);
      setActionStatuses(prev => prev.map(a => a.type === action && a.messageId === messageId ? { ...a, status: 'error', error: errorMessage, phase: undefined, stepIndex: undefined, totalSteps: undefined, note: undefined } : a));
    }
  }, [repoUrl, repoName, repoType, token, selectedProvider, selectedModel, isCustomModel, customModel, language, excludedDirs, excludedFiles, includedDirs, includedFiles, startActionProgress, stopActionProgress]);

  const actionIcon = (action: AgentAction) => {
    if (action === 'GENERATE_PDF') return '📄';
    if (action === 'GENERATE_PPT') return '📊';
    return '🎬';
  };

  // Handle Enter key (submit on Enter, newline on Shift+Enter)
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="min-h-screen bg-[var(--background)] flex flex-col">
      {/* Top navigation bar */}
      <nav className="sticky top-0 z-30 backdrop-blur-md bg-[var(--background)]/80 border-b border-[var(--border-color)]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={() => router.push('/')}
              className="text-[var(--muted)] hover:text-[var(--foreground)] transition-colors"
              title={t?.repoPage?.backToHome || 'Back to Home'}
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
            </button>
            <div className="flex items-center gap-2">
              <svg className="w-5 h-5 text-teal-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
              </svg>
              <span className="text-sm font-semibold text-[var(--foreground)]">{t?.agentChat?.title || 'Agent Chat'}</span>
              <span className="text-xs text-[var(--muted)] hidden sm:inline truncate max-w-[300px]">{repoName || repoUrl}</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {/* Model selection */}
            <button
              onClick={() => setIsModelSelectionModalOpen(true)}
              className="text-xs px-2.5 py-1.5 rounded-md bg-[var(--card-bg)] border border-[var(--border-color)] text-[var(--muted)] hover:text-[var(--foreground)] hover:border-[var(--accent-primary)]/40 transition-colors"
              title={t?.floatingChat?.changeModel || 'Change model'}
            >
              {selectedProvider}/{isCustomModel ? customModel : selectedModel}
            </button>
            {/* Clear */}
            <button
              onClick={clearConversation}
              className="p-2 rounded-md text-[var(--muted)] hover:text-[var(--foreground)] hover:bg-[var(--card-bg)] transition-colors"
              title={t?.floatingChat?.clearChat || 'Clear chat'}
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </button>
            <LanguageToggle />
            <ThemeToggle />
          </div>
        </div>
      </nav>

      {/* Chat area */}
      <main className="flex-1 flex flex-col max-w-4xl w-full mx-auto">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
          {/* Welcome message */}
          {conversationHistory.length === 0 && !streamingResponse && (
            <div className="flex flex-col items-center justify-center py-20 text-center">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-teal-500/20 to-cyan-500/20 flex items-center justify-center mb-4">
                <svg className="w-8 h-8 text-teal-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.184 3.298-1.516 3.298H4.314c-1.7 0-2.748-2.066-1.516-3.298L4.2 15.3" />
                </svg>
              </div>
              <h2 className="text-lg font-semibold text-[var(--foreground)] mb-2">{t?.agentChat?.welcomeTitle || 'Agent Chat'}</h2>
              <p className="text-sm text-[var(--muted)] max-w-md mb-6">
                {t?.agentChat?.welcomeDesc || 'Ask questions about this repository, or ask me to generate a PDF report, PPT presentation, or video overview.'}
              </p>
              {/* Quick action suggestions */}
              <div className="flex flex-wrap gap-2 justify-center">
                {[
                  { label: t?.agentChat?.suggestAnalyze || 'Analyze this repository', query: 'Please analyze this repository and give me an overview of its architecture.' },
                  { label: t?.agentChat?.suggestPdf || 'Generate PDF report', query: 'Please generate a PDF technical report for this repository.' },
                  { label: t?.agentChat?.suggestPpt || 'Generate PPT slides', query: 'Please create a PowerPoint presentation for this repository.' },
                  { label: t?.agentChat?.suggestVideo || 'Generate video overview', query: 'Please generate a video overview of this repository.' },
                ].map((suggestion, i) => (
                  <button
                    key={i}
                    onClick={() => setQuestion(suggestion.query)}
                    className="text-xs px-3 py-1.5 rounded-full bg-[var(--card-bg)] border border-[var(--border-color)] text-[var(--muted)] hover:text-[var(--foreground)] hover:border-teal-500/40 transition-all"
                  >
                    {suggestion.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Conversation messages */}
          {conversationHistory.map((msg) => (
            <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div
                className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm ${
                  msg.role === 'user'
                    ? 'bg-gradient-to-br from-teal-600 to-cyan-500 text-white rounded-br-md'
                    : 'bg-[var(--card-bg)] border border-[var(--border-color)] text-[var(--foreground)] rounded-bl-md'
                }`}
              >
                {msg.role === 'assistant' ? (
                  <>
                    <Markdown content={stripActionTags(msg.content)} />
                    {/* Render action buttons inline */}
                    {parseActions(msg.content).map((action, ai) => {
                      const status = actionStatuses.find(a => a.type === action && a.messageId === msg.id);
                      return (
                        <div key={ai} className="mt-3 pt-3 border-t border-[var(--border-color)]/50">
                          <div className="flex items-center gap-2">
                            <span className="text-base">{actionIcon(action)}</span>
                            {(!status || status.status === 'pending') && (
                              <button
                                onClick={() => executeAction(action, msg.id)}
                                className="text-xs px-3 py-1.5 rounded-md bg-gradient-to-r from-teal-600 to-cyan-500 text-white hover:shadow-md transition-all"
                              >
                                {actionLabel(action)}
                              </button>
                            )}
                            {status?.status === 'running' && (
                              <div className="text-xs text-teal-600 dark:text-teal-400 space-y-1">
                                <div className="flex items-center gap-1.5">
                                  <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                  </svg>
                                  <span>{status.phase || (t?.agentChat?.generating || 'Generating...')}</span>
                                </div>
                                {status.stepIndex && status.totalSteps && (
                                  <div className="text-[11px] text-teal-500/90 dark:text-teal-300/90">
                                    Step {status.stepIndex}/{status.totalSteps}
                                  </div>
                                )}
                                {status.note && (
                                  <div className="text-[11px] text-teal-500/90 dark:text-teal-300/90">
                                    {status.note}
                                  </div>
                                )}
                              </div>
                            )}
                            {status?.status === 'done' && (
                              <span className="text-xs text-green-600 dark:text-green-400">
                                ✓ {t?.agentChat?.downloadReady || 'Downloaded!'}
                              </span>
                            )}
                            {status?.status === 'error' && (
                              <span className="text-xs text-red-500">
                                ✗ {status.error}
                              </span>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </>
                ) : (
                  <span>{getDisplayContent(msg)}</span>
                )}
              </div>
            </div>
          ))}

          {/* Streaming response */}
          {streamingResponse && (
            <div className="flex justify-start">
              <div className="max-w-[80%] rounded-2xl rounded-bl-md px-4 py-3 text-sm bg-[var(--card-bg)] border border-[var(--border-color)] text-[var(--foreground)]">
                <Markdown content={stripActionTags(streamingResponse)} />
              </div>
            </div>
          )}

          {/* Loading indicator */}
          {isLoading && !streamingResponse && (
            <div className="flex justify-start">
              <div className="rounded-2xl rounded-bl-md px-4 py-3 bg-[var(--card-bg)] border border-[var(--border-color)]">
                <div className="flex items-center space-x-2">
                  <div className="flex space-x-1.5">
                    <div className="w-2 h-2 bg-teal-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <div className="w-2 h-2 bg-teal-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <div className="w-2 h-2 bg-teal-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                  <span className="text-xs text-[var(--muted)]">{t?.agentChat?.thinking || 'Thinking...'}</span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Progress is shown inline in the message bubble — no separate status bar needed */}

        {/* Input area */}
        <div className="border-t border-[var(--border-color)] px-4 py-3 bg-[var(--background)]">
          <form onSubmit={handleSubmit} className="max-w-4xl mx-auto space-y-2">
            <div className="flex items-center justify-end">
              <label className="inline-flex items-center gap-2 text-xs text-[var(--muted)] select-none cursor-pointer">
                <input
                  type="checkbox"
                  checked={useDeepResearch}
                  onChange={(e) => setUseDeepResearch(e.target.checked)}
                  className="h-4 w-4 rounded border-[var(--border-color)] text-teal-600 focus:ring-teal-500"
                  disabled={isLoading}
                />
                <span>{t?.ask?.deepResearch || 'Deep Research'}</span>
              </label>
            </div>
            <div className="flex gap-3 items-end">
            <textarea
              ref={inputRef}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={t?.agentChat?.placeholder || 'Ask about this repository, or request a PDF/PPT/Video...'}
              rows={1}
              className="flex-1 resize-none rounded-xl border border-[var(--border-color)] bg-[var(--card-bg)] text-[var(--foreground)] px-4 py-3 text-sm focus:border-teal-500 focus:ring-2 focus:ring-teal-500/20 focus:outline-none transition-all"
              disabled={isLoading}
              style={{ minHeight: '44px', maxHeight: '120px' }}
              onInput={(e) => {
                const target = e.target as HTMLTextAreaElement;
                target.style.height = 'auto';
                target.style.height = Math.min(target.scrollHeight, 120) + 'px';
              }}
            />
            <button
              type="submit"
              disabled={isLoading || !question.trim()}
              className={`px-4 py-3 rounded-xl font-medium text-sm transition-all duration-200 flex-shrink-0 ${
                isLoading || !question.trim()
                  ? 'bg-gray-200 dark:bg-gray-700 text-gray-400 cursor-not-allowed'
                  : 'bg-gradient-to-r from-teal-600 to-cyan-500 text-white hover:shadow-md'
              }`}
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            </button>
            </div>
          </form>
          <p className="text-[10px] text-[var(--muted)] text-center mt-2">
            {t?.agentChat?.hintTools || 'The agent can generate PDF, PPT, and Video when you ask.'}
          </p>
        </div>
      </main>

      {/* Model Selection Modal */}
      <ModelSelectionModal
        isOpen={isModelSelectionModalOpen}
        onClose={() => setIsModelSelectionModalOpen(false)}
        provider={selectedProvider}
        setProvider={setSelectedProvider}
        model={selectedModel}
        setModel={setSelectedModel}
        isCustomModel={isCustomModel}
        setIsCustomModel={setIsCustomModel}
        customModel={customModel}
        setCustomModel={setCustomModel}
        showFileFilters={false}
        onApply={() => {
          console.log('Model selection applied:', selectedProvider, selectedModel);
        }}
        authRequired={false}
        isAuthLoading={false}
      />
    </div>
  );
}

export default function AgentChatPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-[var(--background)] flex items-center justify-center">
        <div className="text-[var(--muted)]">Loading...</div>
      </div>
    }>
      <AgentChatContent />
    </Suspense>
  );
}
