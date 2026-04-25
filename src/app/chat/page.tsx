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
  role: 'user' | 'assistant';
  content: string;
}

// Action types the agent can trigger
type AgentAction = 'GENERATE_PDF' | 'GENERATE_PPT' | 'GENERATE_VIDEO' | 'GENERATE_POSTER' | 'GENERATE_ONBOARD';

interface ActionStatus {
  type: AgentAction;
  status: 'pending' | 'running' | 'done' | 'error';
  phase?: string;
  error?: string;
}

interface ResearchStage {
  title: string;
  content: string;
  iteration: number;
  type: 'plan' | 'update' | 'tool_call' | 'finding' | 'gap' | 'conclusion';
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

  // Deep Research state
  const [researchStages, setResearchStages] = useState<ResearchStage[]>([]);
  const [researchIteration, setResearchIteration] = useState(0);
  const [researchComplete, setResearchComplete] = useState(false);
  const [thinkingExpanded, setThinkingExpanded] = useState(false);

  // Action state — tracks ongoing export actions
  const [actionStatuses, setActionStatuses] = useState<ActionStatus[]>([]);

  const inputRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const webSocketRef = useRef<WebSocket | null>(null);

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

  // ── Detect action tags in assistant response ─────────────
  const parseActions = (content: string): AgentAction[] => {
    const actions: AgentAction[] = [];
    const actionRegex = /\[ACTION:(GENERATE_PDF|GENERATE_PPT|GENERATE_VIDEO|GENERATE_POSTER|GENERATE_ONBOARD)\]/g;
    let match;
    while ((match = actionRegex.exec(content)) !== null) {
      actions.push(match[1] as AgentAction);
    }
    return actions;
  };

  // Strip action tags from display text
  const stripActionTags = (content: string): string => {
    return content.replace(/\[ACTION:(GENERATE_PDF|GENERATE_PPT|GENERATE_VIDEO|GENERATE_POSTER|GENERATE_ONBOARD)\]/g, '').trim();
  };

  // ── Deep Research helpers ──────────────────────────────────
  /**
   * Strip JSON artifacts and tool-status lines that should never appear
   * in the user-visible answer.
   */
  const cleanDisplayText = (text: string): string => {
    if (!text) return text;
    // Remove [RESEARCH_EVENT]{...} markers (with nested braces) using iterative brace-counting
    let cleaned = text;
    const marker = '[RESEARCH_EVENT]';
    let start = cleaned.indexOf(marker);
    while (start !== -1) {
      const jsonStart = start + marker.length;
      if (jsonStart < cleaned.length && cleaned[jsonStart] === '{') {
        let depth = 0, inStr = false, esc = false, end = -1;
        for (let i = jsonStart; i < cleaned.length; i++) {
          const ch = cleaned[i];
          if (esc) { esc = false; continue; }
          if (ch === '\\') { esc = true; continue; }
          if (ch === '"') { inStr = !inStr; continue; }
          if (inStr) continue;
          if (ch === '{') depth++;
          else if (ch === '}') { depth--; if (depth === 0) { end = i + 1; break; } }
        }
        if (end !== -1) {
          cleaned = cleaned.slice(0, start) + cleaned.slice(end);
        } else {
          break;
        }
      } else {
        start = start + marker.length;
      }
      start = cleaned.indexOf(marker, start);
    }
    // Remove <think>...</think> blocks entirely (including content)
    cleaned = cleaned.replace(/<think>[\s\S]*?<\/think>/g, '');
    // Remove unclosed <think> block (still streaming thinking content)
    cleaned = cleaned.replace(/<think>[\s\S]*$/g, '');
    // Remove standalone JSON objects that look like research events
    cleaned = cleaned.replace(/\{"type"\s*:\s*"[^"]*"\s*,\s*"data"\s*:\s*"[^"]*"[^}]*\}/g, '');
    // Remove tool status lines like "> ⚙️ Running ..." or "> 🔍 Searching ..."
    cleaned = cleaned.replace(/^>\s*.{1,4}(?:Running|Searching|Reading|Browsing|Grepping|Finding|\u6b63\u5728).+$/gm, '');
    // Collapse 3+ consecutive blank lines
    cleaned = cleaned.replace(/\n{3,}/g, '\n\n');
    return cleaned.trim();
  };

  /**
   * Extract all [RESEARCH_EVENT]{...} blocks from the stream.
   * displayText comes EXCLUSIVELY from the conclusion event's data field.
   */
  const parseResearchEvents = (raw: string): { displayText: string; stages: ResearchStage[] } => {
    const stages: ResearchStage[] = [];
    let iterationCounter = 0;
    let conclusionText = '';
    const markerStr = '[RESEARCH_EVENT]';
    const markerLen = markerStr.length;

    let searchFrom = 0;
    while (true) {
      const idx = raw.indexOf(markerStr, searchFrom);
      if (idx === -1) break;
      const jsonStart = idx + markerLen;
      if (jsonStart >= raw.length || raw[jsonStart] !== '{') { searchFrom = jsonStart; continue; }

      // Brace-counting to handle nested objects like "metadata": {}
      let depth = 0, inString = false, escaped = false, endPos = -1;
      for (let i = jsonStart; i < raw.length; i++) {
        const ch = raw[i];
        if (escaped) { escaped = false; continue; }
        if (ch === '\\') { escaped = true; continue; }
        if (ch === '"') { inString = !inString; continue; }
        if (inString) continue;
        if (ch === '{') depth++;
        else if (ch === '}') { depth--; if (depth === 0) { endPos = i + 1; break; } }
      }
      if (endPos === -1) { searchFrom = jsonStart; continue; }

      try {
        const json = JSON.parse(raw.slice(jsonStart, endPos));
        const data = (json.data || '').replace(/\\n/g, '\n');
        if (json.type === 'plan') {
          stages.push({ title: '📋 Research Plan', content: data, iteration: 0, type: 'plan' });
        } else if (json.type === 'iteration_start') {
          iterationCounter++;
          stages.push({ title: `🔍 Investigating: ${(json.sub_question || '').slice(0, 60)}`, content: '', iteration: iterationCounter, type: 'update' });
        } else if (json.type === 'tool_call') {
          stages.push({ title: data.slice(0, 80), content: data, iteration: iterationCounter, type: 'tool_call' });
        } else if (json.type === 'finding') {
          const lastUpdate = [...stages].reverse().find(s => s.type === 'update');
          if (lastUpdate) { lastUpdate.content += data; }
          else { stages.push({ title: 'Finding', content: data, iteration: iterationCounter, type: 'finding' }); }
        } else if (json.type === 'iteration_end') {
          const lastUpdate = [...stages].reverse().find(s => s.type === 'update' && s.iteration === iterationCounter);
          if (lastUpdate && !lastUpdate.content) { lastUpdate.content = data; }
        } else if (json.type === 'gap_analysis') {
          stages.push({ title: '💡 New aspects to explore', content: data, iteration: iterationCounter, type: 'gap' });
        } else if (json.type === 'synthesis_start') {
          stages.push({ title: '📝 Synthesizing final answer...', content: '', iteration: iterationCounter + 1, type: 'conclusion' });
        } else if (json.type === 'conclusion') {
          conclusionText = data;
        }
      } catch { /* ignore malformed */ }

      searchFrom = endPos;
    }

    return { displayText: conclusionText, stages };
  };

  // ── Execute an export action ─────────────────────────────
  const executeAction = useCallback(async (action: AgentAction) => {
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
    } else if (action === 'GENERATE_POSTER') {
      endpoint = '/api/export/repo-poster';
      defaultFilename = `${repoName.split('/').pop() || 'repo'}_poster.png`;
    } else if (action === 'GENERATE_ONBOARD') {
      endpoint = '/api/export/repo-onboard';
      // Onboard returns JSON; no file download. Handled in special branch below.
    }

    setActionStatuses(prev => [...prev.filter(a => a.type !== action), { type: action, status: 'running', phase: 'Generating...' }]);

    // ── Onboard: render markdown inline as a new chat message instead of downloading
    if (action === 'GENERATE_ONBOARD') {
      try {
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(bodyPayload),
        });
        if (!response.ok) {
          const errorText = await response.text().catch(() => 'Unknown error');
          throw new Error(`Onboard failed: ${response.status} - ${errorText}`);
        }
        const data = await response.json();
        const markdown = (data && typeof data.markdown === 'string') ? data.markdown : '';
        if (markdown) {
          // Append as a new assistant message so the chat renders it.
          setConversationHistory(prev => [...prev, { role: 'assistant', content: markdown }]);
        }
        setActionStatuses(prev => prev.map(a => a.type === action ? { ...a, status: 'done', phase: undefined } : a));
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Unknown error';
        setActionStatuses(prev => prev.map(a => a.type === action ? { ...a, status: 'error', error: errorMessage, phase: undefined } : a));
      }
      return;
    }

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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

      setActionStatuses(prev => prev.map(a => a.type === action ? { ...a, status: 'done', phase: undefined } : a));
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setActionStatuses(prev => prev.map(a => a.type === action ? { ...a, status: 'error', error: errorMessage, phase: undefined } : a));
    }
  }, [repoUrl, repoName, repoType, token, selectedProvider, selectedModel, isCustomModel, customModel, language, excludedDirs, excludedFiles, includedDirs, includedFiles]);

  // ── Handle chat submit ───────────────────────────────────
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || isLoading) return;

    const modePrefix = useDeepResearch ? '[AGENT] [DEEP RESEARCH]' : '[AGENT]';
    const actualContent = `${modePrefix} ${question}`;
    const userMessage: Message = { role: 'user', content: actualContent };
    const newHistory = [...conversationHistory, userMessage];
    setConversationHistory(newHistory);
    setQuestion('');
    setIsLoading(true);
    setStreamingResponse('');

    // Reset research state for new deep research question
    if (useDeepResearch) {
      setResearchIteration(0);
      setResearchComplete(false);
      setResearchStages([]);
      setThinkingExpanded(false);
    }

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
        if (useDeepResearch) {
          const { displayText, stages } = parseResearchEvents(fullResponse);
          setStreamingResponse(cleanDisplayText(displayText));
          if (stages.length > 0) { setResearchStages(stages); setResearchIteration(stages.length); }
        } else {
          setStreamingResponse(cleanDisplayText(fullResponse));
        }
      },
      (error: Event) => {
        console.error('WebSocket error:', error);
        fullResponse += '\n\nError: Connection failed. Trying HTTP fallback...';
        setStreamingResponse(fullResponse);
        fallbackToHttp(requestBody, newHistory);
      },
      () => {
        if (fullResponse) {
          const displayContent = useDeepResearch
            ? cleanDisplayText(parseResearchEvents(fullResponse).displayText)
            : cleanDisplayText(fullResponse);
          setConversationHistory(prev => [...prev, { role: 'assistant', content: displayContent }]);
          setStreamingResponse('');
          // Detect actions in response
          const actions = parseActions(displayContent);
          if (actions.length > 0) {
            actions.forEach(action => executeAction(action));
          }
        }
        if (useDeepResearch) {
          setResearchComplete(true);
          setThinkingExpanded(false);
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

        if (useDeepResearch) {
          const { displayText, stages } = parseResearchEvents(fullResponse);
          setStreamingResponse(cleanDisplayText(displayText));
          if (stages.length > 0) { setResearchStages(stages); setResearchIteration(stages.length); }
        } else {
          setStreamingResponse(fullResponse);
        }
      }

      const displayContent = useDeepResearch
        ? cleanDisplayText(parseResearchEvents(fullResponse).displayText)
        : fullResponse;
      setConversationHistory([...history, { role: 'assistant', content: displayContent }]);
      setStreamingResponse('');
      const actions = parseActions(displayContent);
      if (actions.length > 0) {
        actions.forEach(action => executeAction(action));
      }
      if (useDeepResearch) {
        setResearchComplete(true);
        setThinkingExpanded(false);
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
    setResearchIteration(0);
    setResearchComplete(false);
    setResearchStages([]);
    setThinkingExpanded(false);
  };

  // Strip [AGENT] tag and research events from display
  const getDisplayContent = (msg: Message): string => {
    let content = msg.content.replace(/^\[AGENT\]\s*(\[DEEP RESEARCH\]\s*)?/, '');
    if (content.includes('[RESEARCH_EVENT]')) {
      const { displayText } = parseResearchEvents(content);
      return cleanDisplayText(displayText);
    }
    return cleanDisplayText(content);
  };

  // Action label mapping
  const actionLabel = (action: AgentAction): string => {
    const labels: Record<AgentAction, string> = {
      GENERATE_PDF: t?.agentChat?.generatingPdf || 'PDF Report',
      GENERATE_PPT: t?.agentChat?.generatingPpt || 'PPT Slides',
      GENERATE_VIDEO: t?.agentChat?.generatingVideo || 'Video Overview',
      GENERATE_POSTER: t?.agentChat?.generatingPoster || 'Poster',
      GENERATE_ONBOARD: t?.agentChat?.generatingOnboard || 'Beginner Quick Start',
    };
    return labels[action];
  };

  const actionIcon = (action: AgentAction) => {
    if (action === 'GENERATE_PDF') return '📄';
    if (action === 'GENERATE_PPT') return '📊';
    if (action === 'GENERATE_POSTER') return '🖼️';
    if (action === 'GENERATE_ONBOARD') return '🚀';
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
                  { label: t?.agentChat?.suggestPoster || 'Generate poster', query: 'Please generate an illustrated poster for this repository.' },
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
          {conversationHistory.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
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
                      const status = actionStatuses.find(a => a.type === action);
                      return (
                        <div key={ai} className="mt-3 pt-3 border-t border-[var(--border-color)]/50">
                          <div className="flex items-center gap-2">
                            <span className="text-base">{actionIcon(action)}</span>
                            {(!status || status.status === 'pending') && (
                              <button
                                onClick={() => executeAction(action)}
                                className="text-xs px-3 py-1.5 rounded-md bg-gradient-to-r from-teal-600 to-cyan-500 text-white hover:shadow-md transition-all"
                              >
                                {actionLabel(action)}
                              </button>
                            )}
                            {status?.status === 'running' && (
                              <span className="text-xs text-teal-600 dark:text-teal-400 flex items-center gap-1.5">
                                <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24">
                                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                </svg>
                                {status.phase || (t?.agentChat?.generating || 'Generating...')}
                              </span>
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
          {(streamingResponse || (useDeepResearch && researchStages.length > 0 && isLoading)) && (
            <div className="flex justify-start">
              <div className="max-w-[80%] rounded-2xl rounded-bl-md px-4 py-3 text-sm bg-[var(--card-bg)] border border-[var(--border-color)] text-[var(--foreground)]">
                {/* Collapsible research thinking section */}
                {useDeepResearch && researchStages.length > 0 && (
                  <div className="mb-3">
                    <button
                      onClick={() => setThinkingExpanded(!thinkingExpanded)}
                      className="flex items-center gap-1.5 text-xs text-[var(--muted)] hover:text-[var(--foreground)] transition-colors"
                    >
                      <svg
                        className={`w-3 h-3 transition-transform ${thinkingExpanded ? 'rotate-90' : ''}`}
                        fill="none" viewBox="0 0 24 24" stroke="currentColor"
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                      </svg>
                      <span>🔬</span>
                      <span className="font-medium">Research Process</span>
                      <span className="text-[var(--muted)]">
                        ({researchStages.filter(s => s.type === 'update').length} iterations)
                      </span>
                      {isLoading && !researchComplete && (
                        <span className="inline-block w-1.5 h-1.5 bg-teal-500 rounded-full animate-pulse" />
                      )}
                    </button>
                    {thinkingExpanded && (
                      <div className="mt-1.5 pl-4 border-l-2 border-[var(--border-color)] max-h-[250px] overflow-y-auto">
                        <div className="space-y-1.5">
                          {researchStages.map((stage, idx) => (
                            <div key={idx} className="flex items-start gap-2 text-xs">
                              <div className={`mt-1 w-1.5 h-1.5 rounded-full flex-shrink-0 ${
                                stage.type === 'plan' ? 'bg-blue-500' :
                                stage.type === 'update' ? 'bg-teal-500' :
                                stage.type === 'tool_call' ? 'bg-gray-400' :
                                stage.type === 'gap' ? 'bg-amber-500' :
                                stage.type === 'conclusion' ? 'bg-green-500' :
                                'bg-gray-400'
                              }`} />
                              <div className="min-w-0">
                                <span className="text-[var(--foreground)]">{stage.title}</span>
                                {stage.content && stage.type !== 'tool_call' && (
                                  <p className="text-[var(--muted)] mt-0.5 line-clamp-2">{stage.content.slice(0, 150)}</p>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
                {streamingResponse && <Markdown content={stripActionTags(streamingResponse)} />}
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

        {/* Active actions status bar */}
        {actionStatuses.some(a => a.status === 'running') && (
          <div className="px-4 py-2 border-t border-[var(--border-color)] bg-teal-50 dark:bg-teal-900/20">
            <div className="flex items-center gap-3 text-xs text-teal-700 dark:text-teal-300">
              <svg className="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              {actionStatuses.filter(a => a.status === 'running').map(a => actionLabel(a.type)).join(', ')} — {t?.agentChat?.exportInProgress || 'Export in progress...'}
            </div>
          </div>
        )}

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
