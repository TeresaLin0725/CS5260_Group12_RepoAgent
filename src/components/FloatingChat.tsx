'use client';

import React, { useState, useRef, useEffect } from 'react';
import { FaChevronLeft, FaChevronRight } from 'react-icons/fa';
import Markdown from './Markdown';
import ModelSelectionModal from './ModelSelectionModal';
import { useLanguage } from '@/contexts/LanguageContext';
import { createChatWebSocket, closeWebSocket, ChatCompletionRequest } from '@/utils/websocketClient';

interface Model {
  id: string;
  name: string;
}

interface Provider {
  id: string;
  name: string;
  models: Model[];
  supportsCustomModel?: boolean;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface ResearchStage {
  title: string;
  content: string;
  iteration: number;
  type: 'plan' | 'update' | 'conclusion';
}

interface FloatingChatProps {
  repoUrl: string;
  repoType?: string;
  token?: string;
  provider?: string;
  model?: string;
  isCustomModel?: boolean;
  customModel?: string;
  language?: string;
}

const FloatingChat: React.FC<FloatingChatProps> = ({
  repoUrl,
  repoType = 'github',
  token,
  provider = '',
  model = '',
  isCustomModel = false,
  customModel = '',
  language = 'en',
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [question, setQuestion] = useState('');
  const [streamingResponse, setStreamingResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationHistory, setConversationHistory] = useState<Message[]>([]);

  // Deep Research state
  const [deepResearch, setDeepResearch] = useState(false);
  const [researchIteration, setResearchIteration] = useState(0);
  const [researchComplete, setResearchComplete] = useState(false);
  const [researchStages, setResearchStages] = useState<ResearchStage[]>([]);
  const [currentStageIndex, setCurrentStageIndex] = useState(0);

  // Model selection state
  const [selectedProvider, setSelectedProvider] = useState(provider);
  const [selectedModel, setSelectedModel] = useState(model);
  const [isCustomSelectedModel, setIsCustomSelectedModel] = useState(isCustomModel);
  const [customSelectedModel, setCustomSelectedModel] = useState(customModel);
  const [isModelSelectionModalOpen, setIsModelSelectionModalOpen] = useState(false);

  const { messages: i18n } = useLanguage();
  const inputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const webSocketRef = useRef<WebSocket | null>(null);

  // Refs to avoid stale closures in continueResearch / effects
  const conversationHistoryRef = useRef(conversationHistory);
  conversationHistoryRef.current = conversationHistory;
  const researchIterationRef = useRef(researchIteration);
  researchIterationRef.current = researchIteration;
  const researchCompleteRef = useRef(researchComplete);
  researchCompleteRef.current = researchComplete;
  const isLoadingRef = useRef(isLoading);
  isLoadingRef.current = isLoading;
  const deepResearchRef = useRef(deepResearch);
  deepResearchRef.current = deepResearch;

  // Fetch default model config if no provider/model supplied
  useEffect(() => {
    const fetchModel = async () => {
      try {
        const response = await fetch('/api/models/config');
        if (!response.ok) return;
        const data = await response.json();
        if (!provider && !model) {
          setSelectedProvider(data.defaultProvider);
          const defaultProv = data.providers.find((p: Provider) => p.id === data.defaultProvider);
          if (defaultProv && defaultProv.models.length > 0) {
            setSelectedModel(defaultProv.models[0].id);
          }
        }
      } catch (err) {
        console.error('Failed to fetch model configurations:', err);
      }
    };
    if (!provider && !model) fetchModel();
  }, [provider, model]);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversationHistory, streamingResponse]);

  // Focus input when opened
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  // Cleanup websocket on unmount
  useEffect(() => {
    return () => { closeWebSocket(webSocketRef.current); };
  }, []);

  // ── Deep Research helpers ──────────────────────────────────

  const checkIfResearchComplete = (content: string): boolean => {
    // Explicit final conclusion header — always terminal
    if (content.includes('## Final Conclusion')) return true;

    // Phrases that clearly signal continuation — if ANY is present, not done
    const continuationSignals = [
      'I will now proceed to',
      'Next Steps',
      'next iteration',
      'final iteration',              // "In the final iteration, I will…"
      'In the next',                   // "In the next iteration / section"
      'I will focus on',
      'I will synthesize',
      'I will delve',
      'I will explore',
      'I will analyze',
      'further investigation',
      'further research',
      'remaining questions',
      'deeper into',
    ];
    const hasContinuation = continuationSignals.some(s => content.includes(s));

    // "## Conclusion" / "## Summary" is only terminal when there is NO continuation language
    if ((content.includes('## Conclusion') || content.includes('## Summary')) && !hasContinuation) {
      return true;
    }

    // Other explicit completion phrases — still gated by no-continuation
    if (!hasContinuation && (
      content.includes('This concludes our research') ||
      content.includes('This completes our investigation') ||
      content.includes('This concludes the deep research process') ||
      content.includes('Key Findings and Implementation Details')
    )) return true;

    return false;
  };

  const extractResearchStage = (content: string, iteration: number): ResearchStage | null => {
    if (iteration === 1 && content.includes('## Research Plan')) {
      return { title: 'Research Plan', content, iteration: 1, type: 'plan' };
    }
    if (iteration >= 1 && iteration <= 4) {
      if (content.match(new RegExp(`## Research Update ${iteration}`))) {
        return { title: `Research Update ${iteration}`, content, iteration, type: 'update' };
      }
    }
    if (content.includes('## Final Conclusion')) {
      return { title: 'Final Conclusion', content, iteration, type: 'conclusion' };
    }
    return null;
  };

  const upsertStage = (stage: ResearchStage) => {
    setResearchStages(prev => {
      const idx = prev.findIndex(s => s.iteration === stage.iteration && s.type === stage.type);
      if (idx >= 0) { const n = [...prev]; n[idx] = stage; return n; }
      return [...prev, stage];
    });
  };

  // Navigate stages
  const navigateToStage = (index: number) => {
    if (index >= 0 && index < researchStages.length) setCurrentStageIndex(index);
  };

  // Auto-continue deep research (reads latest state via refs to avoid stale closures)
  const continueResearch = async () => {
    if (!deepResearchRef.current || researchCompleteRef.current || isLoadingRef.current) return;

    setIsLoading(true);
    setStreamingResponse('');

    const continueMsg: Message = { role: 'user', content: '[DEEP RESEARCH] Continue the research' };
    // Build from ref (synchronously up-to-date) — setState functional updater is async and would leave local var empty
    const newHistory = [...conversationHistoryRef.current, continueMsg];
    setConversationHistory(newHistory);

    const newIteration = researchIterationRef.current + 1;
    setResearchIteration(newIteration);

    console.log(`[DeepResearch] continueResearch fired — iteration ${newIteration}, messages: ${newHistory.length}`);

    const requestBody: ChatCompletionRequest = {
      repo_url: repoUrl,
      type: repoType,
      messages: newHistory.map(msg => ({ role: msg.role, content: msg.content })),
      provider: selectedProvider,
      model: isCustomSelectedModel ? customSelectedModel : selectedModel,
      language: language,
    };
    if (token) requestBody.token = token;

    closeWebSocket(webSocketRef.current);
    let fullResponse = '';

    webSocketRef.current = createChatWebSocket(
      requestBody,
      (message: string) => {
        fullResponse += message;
        setStreamingResponse(fullResponse);
        if (deepResearchRef.current) {
          const stage = extractResearchStage(fullResponse, newIteration);
          if (stage) { upsertStage(stage); setCurrentStageIndex(researchStages.length); }
        }
      },
      (error: Event) => {
        console.error('WebSocket error:', error);
        fullResponse += '\n\nError: Connection failed. Trying HTTP fallback...';
        setStreamingResponse(fullResponse);
        fallbackToHttp(requestBody, newHistory, newIteration);
      },
      () => {
        if (fullResponse) {
          setConversationHistory(prev => [...prev, { role: 'assistant', content: fullResponse }]);
          setStreamingResponse('');
        }
        const isComplete = checkIfResearchComplete(fullResponse);
        const forceComplete = newIteration >= 5;
        if (forceComplete && !isComplete) {
          const note = '\n\n## Final Conclusion\nAfter multiple iterations of deep research, we\'ve gathered significant insights. This concludes the research process.';
          setConversationHistory(prev => {
            const updated = [...prev];
            if (updated.length > 0 && updated[updated.length - 1].role === 'assistant') {
              updated[updated.length - 1] = { role: 'assistant', content: updated[updated.length - 1].content + note };
            }
            return updated;
          });
          setResearchComplete(true);
        } else {
          setResearchComplete(isComplete);
        }
        setIsLoading(false);
      }
    );
  };

  // Auto-continue effect
  useEffect(() => {
    if (!deepResearch || isLoading || researchComplete) return;
    if (researchIteration <= 0 || researchIteration >= 5) return;

    const lastMsg = conversationHistory[conversationHistory.length - 1];
    if (lastMsg?.role !== 'assistant') return;

    if (checkIfResearchComplete(lastMsg.content)) {
      setResearchComplete(true);
      return;
    }

    console.log(`[DeepResearch] scheduling continue — iteration ${researchIteration}, history len ${conversationHistory.length}`);
    const timer = setTimeout(() => { continueResearch(); }, 2000);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversationHistory, isLoading, deepResearch, researchComplete, researchIteration]);

  // Research stage extraction effect
  useEffect(() => {
    if (deepResearch && !isLoading && conversationHistory.length > 0) {
      const lastMsg = conversationHistory[conversationHistory.length - 1];
      if (lastMsg?.role === 'assistant') {
        const stage = extractResearchStage(lastMsg.content, researchIteration || 1);
        if (stage) {
          upsertStage(stage);
          setCurrentStageIndex(prev => {
            const idx = researchStages.findIndex(s => s.iteration === stage.iteration && s.type === stage.type);
            return idx >= 0 ? idx : prev;
          });
        }
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversationHistory, isLoading, deepResearch, researchIteration]);

  // Display helpers
  const getDisplayContent = (msg: Message): string => {
    return msg.role === 'user' ? msg.content.replace(/^\[DEEP RESEARCH\]\s*/, '') : msg.content;
  };
  const isAutoResearchContinue = (msg: Message): boolean => {
    return msg.role === 'user' && msg.content === '[DEEP RESEARCH] Continue the research';
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || isLoading) return;

    const actualContent = deepResearch ? `[DEEP RESEARCH] ${question}` : question;
    const userMessage: Message = { role: 'user', content: actualContent };
    const newHistory = [...conversationHistory, userMessage];
    setConversationHistory(newHistory);
    setQuestion('');
    setIsLoading(true);
    setStreamingResponse('');

    // Reset research state for new question
    if (deepResearch) {
      setResearchIteration(0);
      setResearchComplete(false);
      setResearchStages([]);
      setCurrentStageIndex(0);
    }

    const requestBody: ChatCompletionRequest = {
      repo_url: repoUrl,
      type: repoType,
      messages: newHistory.map(msg => ({ role: msg.role, content: msg.content })),
      provider: selectedProvider,
      model: isCustomSelectedModel ? customSelectedModel : selectedModel,
      language: language,
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
          setConversationHistory(prev => [...prev, { role: 'assistant', content: fullResponse }]);
          setStreamingResponse('');
        }
        // Deep research: check if we need to continue
        if (deepResearch) {
          const isComplete = checkIfResearchComplete(fullResponse);
          if (!isComplete) {
            setResearchIteration(1);
          } else {
            setResearchComplete(true);
          }
        }
        setIsLoading(false);
      }
    );
  };

  const fallbackToHttp = async (requestBody: ChatCompletionRequest, history: Message[], iteration?: number) => {
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

        // Extract research stages during HTTP streaming
        if (deepResearch) {
          const iter = iteration ?? researchIteration;
          const stage = extractResearchStage(fullResponse, iter || 1);
          if (stage) upsertStage(stage);
        }
      }

      setConversationHistory([...history, { role: 'assistant', content: fullResponse }]);
      setStreamingResponse('');

      // Deep research completion check for HTTP path
      if (deepResearch) {
        const isComplete = checkIfResearchComplete(fullResponse);
        const iter = iteration ?? researchIteration;
        const forceComplete = iter >= 5;
        if (forceComplete && !isComplete) {
          const note = '\n\n## Final Conclusion\nAfter multiple iterations of deep research, we\'ve gathered significant insights. This concludes the research process.';
          setConversationHistory(prev => {
            const updated = [...prev];
            if (updated.length > 0 && updated[updated.length - 1].role === 'assistant') {
              updated[updated.length - 1] = { role: 'assistant', content: updated[updated.length - 1].content + note };
            }
            return updated;
          });
          setResearchComplete(true);
        } else if (!isComplete && !iteration) {
          setResearchIteration(1);
        } else {
          setResearchComplete(isComplete);
        }
      }
    } catch (error) {
      console.error('HTTP fallback error:', error);
      setStreamingResponse(prev => prev + '\n\nError: Failed to get response.');
      if (deepResearch) setResearchComplete(true);
    } finally {
      setIsLoading(false);
    }
  };

  const clearConversation = () => {
    setConversationHistory([]);
    setStreamingResponse('');
    setQuestion('');
    setResearchIteration(0);
    setResearchComplete(false);
    setResearchStages([]);
    setCurrentStageIndex(0);
  };

  const floatingChatMessages = i18n.floatingChat;

  return (
    <>
      {/* Floating bubble button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="fixed bottom-6 right-6 z-50 w-14 h-14 rounded-full bg-gradient-to-br from-teal-600 to-cyan-500 text-white shadow-lg hover:shadow-xl hover:scale-105 transition-all duration-200 flex items-center justify-center group"
        title={floatingChatMessages?.toggleChat || 'AI Assistant'}
      >
        {isOpen ? (
          <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        ) : (
          <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
          </svg>
        )}
        {/* Pulse animation when closed */}
        {!isOpen && (
          <span className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 rounded-full animate-pulse" />
        )}
      </button>

      {/* Chat panel */}
      {isOpen && (
        <div className="fixed bottom-24 right-6 z-50 w-[420px] max-w-[calc(100vw-2rem)] h-[600px] max-h-[calc(100vh-8rem)] bg-[var(--card-bg,#fff)] dark:bg-[var(--card-bg,#1e1e2e)] border border-[var(--border-color,#e5e7eb)] dark:border-[var(--border-color,#374151)] rounded-2xl shadow-2xl flex flex-col overflow-hidden">
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 bg-gradient-to-r from-teal-600 to-cyan-500 text-white">
            <div className="flex items-center gap-2">
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.184 3.298-1.516 3.298H4.314c-1.7 0-2.748-2.066-1.516-3.298L4.2 15.3" />
              </svg>
              <span className="font-semibold text-sm">{floatingChatMessages?.title || 'AI Assistant'}</span>
            </div>
            <div className="flex items-center gap-1">
              {/* Model selection */}
              <button
                onClick={() => setIsModelSelectionModalOpen(true)}
                className="text-xs px-2 py-1 rounded bg-white/20 hover:bg-white/30 transition-colors"
                title={floatingChatMessages?.changeModel || 'Change model'}
              >
                {selectedProvider}/{isCustomSelectedModel ? customSelectedModel : selectedModel}
              </button>
              {/* Clear */}
              <button
                onClick={clearConversation}
                className="p-1.5 rounded hover:bg-white/20 transition-colors"
                title={floatingChatMessages?.clearChat || 'Clear chat'}
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              </button>
              {/* Close */}
              <button
                onClick={() => setIsOpen(false)}
                className="p-1.5 rounded hover:bg-white/20 transition-colors"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>

          {/* Messages area */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {conversationHistory.length === 0 && !streamingResponse && (
              <div className="flex flex-col items-center justify-center h-full text-center text-gray-400 dark:text-gray-500">
                <svg className="w-12 h-12 mb-3 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                </svg>
                <p className="text-sm">{floatingChatMessages?.emptyState || 'Ask anything about this repository!'}</p>
              </div>
            )}

            {conversationHistory.map((msg, idx) => (
              isAutoResearchContinue(msg) ? (
                <div key={idx} className="flex justify-center">
                  <span className="text-xs text-teal-600 dark:text-teal-400 bg-teal-50 dark:bg-teal-900/30 px-3 py-1 rounded-full">
                    {floatingChatMessages?.researchContinuing || 'Continuing research…'}
                  </span>
                </div>
              ) : (
                <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div
                    className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm ${
                      msg.role === 'user'
                        ? 'bg-gradient-to-br from-teal-600 to-cyan-500 text-white rounded-br-md'
                        : 'bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200 rounded-bl-md'
                    }`}
                  >
                    {msg.role === 'assistant' ? (
                      <Markdown content={msg.content} />
                    ) : (
                      <span>{getDisplayContent(msg)}</span>
                    )}
                  </div>
                </div>
              )
            ))}

            {/* Streaming response */}
            {streamingResponse && (
              <div className="flex justify-start">
                <div className="max-w-[85%] rounded-2xl rounded-bl-md px-4 py-2.5 text-sm bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200">
                  <Markdown content={streamingResponse} />
                </div>
              </div>
            )}

            {/* Loading indicator */}
            {isLoading && !streamingResponse && (
              <div className="flex justify-start">
                <div className="rounded-2xl rounded-bl-md px-4 py-3 bg-gray-100 dark:bg-gray-800">
                  <div className="flex items-center space-x-2">
                    <div className="flex space-x-1.5">
                      <div className="w-2 h-2 bg-teal-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                      <div className="w-2 h-2 bg-teal-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                      <div className="w-2 h-2 bg-teal-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                    {deepResearch && (
                      <span className="text-xs text-teal-600 dark:text-teal-400">
                        {researchIteration === 0
                          ? (floatingChatMessages?.researchPlanning || 'Planning research…')
                          : `${floatingChatMessages?.researchIteration || 'Iteration'} ${researchIteration}…`}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Stage navigation bar */}
          {deepResearch && researchStages.length > 1 && (
            <div className="flex items-center justify-center gap-2 px-3 py-1.5 border-t border-[var(--border-color,#e5e7eb)] dark:border-[var(--border-color,#374151)] bg-gray-50 dark:bg-gray-800/50">
              <button
                onClick={() => navigateToStage(currentStageIndex - 1)}
                disabled={currentStageIndex === 0}
                className={`p-1 rounded ${
                  currentStageIndex === 0
                    ? 'text-gray-300 dark:text-gray-600'
                    : 'text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700'
                }`}
              >
                <FaChevronLeft size={10} />
              </button>
              <span className="text-xs text-gray-500 dark:text-gray-400">
                {currentStageIndex + 1} / {researchStages.length}
                <span className="ml-1.5">{researchStages[currentStageIndex]?.title}</span>
              </span>
              <button
                onClick={() => navigateToStage(currentStageIndex + 1)}
                disabled={currentStageIndex === researchStages.length - 1}
                className={`p-1 rounded ${
                  currentStageIndex === researchStages.length - 1
                    ? 'text-gray-300 dark:text-gray-600'
                    : 'text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700'
                }`}
              >
                <FaChevronRight size={10} />
              </button>
            </div>
          )}

          {/* Input area */}
          <div className="border-t border-[var(--border-color,#e5e7eb)] dark:border-[var(--border-color,#374151)] p-3">
            {/* Deep Research toggle */}
            <div className="flex items-center justify-between mb-2">
              <div className="group relative">
                <label className="flex items-center cursor-pointer">
                  <span className="text-xs text-gray-500 dark:text-gray-400 mr-1.5">Deep Research</span>
                  <div className="relative">
                    <input
                      type="checkbox"
                      checked={deepResearch}
                      onChange={() => setDeepResearch(!deepResearch)}
                      className="sr-only"
                    />
                    <div className={`w-8 h-4 rounded-full transition-colors ${deepResearch ? 'bg-teal-600' : 'bg-gray-300 dark:bg-gray-600'}`} />
                    <div className={`absolute left-0.5 top-0.5 w-3 h-3 rounded-full bg-white transition-transform ${deepResearch ? 'translate-x-4' : ''}`} />
                  </div>
                </label>
                {/* Tooltip */}
                <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block bg-gray-800 text-white text-xs rounded p-2 w-56 z-10">
                  <div className="relative">
                    <div className="absolute -bottom-2 left-4 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-800" />
                    <p>{floatingChatMessages?.researchTooltip || 'Multi-turn deep investigation (up to 5 iterations). The AI auto-continues until a final conclusion is reached.'}</p>
                  </div>
                </div>
              </div>
              {deepResearch && researchIteration > 0 && (
                <span className="text-xs text-teal-600 dark:text-teal-400">
                  {researchComplete
                    ? (floatingChatMessages?.researchDone || 'Research complete')
                    : `${floatingChatMessages?.researchIteration || 'Iteration'} ${researchIteration}/5`}
                </span>
              )}
            </div>
            <form onSubmit={handleSubmit} className="flex gap-2">
              <input
                ref={inputRef}
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder={floatingChatMessages?.placeholder || 'Ask about this repository...'}
                className="flex-1 rounded-xl border border-[var(--border-color,#e5e7eb)] dark:border-[var(--border-color,#374151)] bg-[var(--input-bg,#fff)] dark:bg-[var(--input-bg,#111827)] text-[var(--foreground)] px-4 py-2.5 text-sm focus:border-teal-500 focus:ring-2 focus:ring-teal-500/20 focus:outline-none transition-all"
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={isLoading || !question.trim()}
                className={`px-4 py-2.5 rounded-xl font-medium text-sm transition-all duration-200 ${
                  isLoading || !question.trim()
                    ? 'bg-gray-200 dark:bg-gray-700 text-gray-400 cursor-not-allowed'
                    : 'bg-gradient-to-r from-teal-600 to-cyan-500 text-white hover:shadow-md'
                }`}
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              </button>
            </form>
          </div>
        </div>
      )}

      {/* Model Selection Modal */}
      <ModelSelectionModal
        isOpen={isModelSelectionModalOpen}
        onClose={() => setIsModelSelectionModalOpen(false)}
        provider={selectedProvider}
        setProvider={setSelectedProvider}
        model={selectedModel}
        setModel={setSelectedModel}
        isCustomModel={isCustomSelectedModel}
        setIsCustomModel={setIsCustomSelectedModel}
        customModel={customSelectedModel}
        setCustomModel={setCustomSelectedModel}
        showFileFilters={false}
        onApply={() => {
          console.log('Model selection applied:', selectedProvider, selectedModel);
        }}
        authRequired={false}
        isAuthLoading={false}
      />
    </>
  );
};

export default FloatingChat;
