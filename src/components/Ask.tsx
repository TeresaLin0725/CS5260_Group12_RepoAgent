'use client';

import React, {useState, useRef, useEffect} from 'react';

import Markdown from './Markdown';
import { useLanguage } from '@/contexts/LanguageContext';
import RepoInfo from '@/types/repoinfo';
import getRepoUrl from '@/utils/getRepoUrl';
import ModelSelectionModal from './ModelSelectionModal';
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
  role: 'user' | 'assistant' | 'system';
  content: string;
}

interface ResearchStage {
  title: string;
  content: string;
  iteration: number;
  type: 'plan' | 'update' | 'tool_call' | 'finding' | 'gap' | 'conclusion';
}

interface AskProps {
  repoInfo: RepoInfo;
  provider?: string;
  model?: string;
  isCustomModel?: boolean;
  customModel?: string;
  language?: string;
  onRef?: (ref: { clearConversation: () => void }) => void;
}

/** Generate or retrieve a stable user ID from localStorage. */
function getOrCreateUserId(): string {
  if (typeof window === 'undefined') return 'anonymous';
  const key = 'deepwiki_user_id';
  let uid = localStorage.getItem(key);
  if (!uid) {
    uid = 'u_' + Math.random().toString(36).slice(2, 11) + Date.now().toString(36);
    localStorage.setItem(key, uid);
  }
  return uid;
}

const Ask: React.FC<AskProps> = ({
  repoInfo,
  provider = '',
  model = '',
  isCustomModel = false,
  customModel = '',
  language = 'en',
  onRef
}) => {
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [deepResearch, setDeepResearch] = useState(false);
  const [userId, setUserId] = useState<string>('anonymous');

  // Model selection state
  const [selectedProvider, setSelectedProvider] = useState(provider);
  const [selectedModel, setSelectedModel] = useState(model);
  const [isCustomSelectedModel, setIsCustomSelectedModel] = useState(isCustomModel);
  const [customSelectedModel, setCustomSelectedModel] = useState(customModel);
  const [isModelSelectionModalOpen, setIsModelSelectionModalOpen] = useState(false);

  // Get language context for translations
  const { messages } = useLanguage();

  // Research state
  const [researchStages, setResearchStages] = useState<ResearchStage[]>([]);
  const [conversationHistory, setConversationHistory] = useState<Message[]>([]);
  const [researchIteration, setResearchIteration] = useState(0);
  const [researchComplete, setResearchComplete] = useState(false);
  const [thinkingExpanded, setThinkingExpanded] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const responseRef = useRef<HTMLDivElement>(null);
  const providerRef = useRef(provider);
  const modelRef = useRef(model);

  // Initialize persistent user ID from localStorage
  useEffect(() => {
    setUserId(getOrCreateUserId());
  }, []);

  // Focus input on component mount
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, []);

  // Expose clearConversation method to parent component
  useEffect(() => {
    if (onRef) {
      onRef({ clearConversation });
    }
  }, [onRef]);

  // Scroll to bottom of response when it changes
  useEffect(() => {
    if (responseRef.current) {
      responseRef.current.scrollTop = responseRef.current.scrollHeight;
    }
  }, [response]);

  // Close WebSocket when component unmounts
  useEffect(() => {
    return () => {
      closeWebSocket(webSocketRef.current);
    };
  }, []);

  useEffect(() => {
    providerRef.current = provider;
    modelRef.current = model;
  }, [provider, model]);

  useEffect(() => {
    const fetchModel = async () => {
      try {
        setIsLoading(true);

        const response = await fetch('/api/models/config');
        if (!response.ok) {
          throw new Error(`Error fetching model configurations: ${response.status}`);
        }

        const data = await response.json();

        // use latest provider/model ref to check
        if(providerRef.current == '' || modelRef.current== '') {
          setSelectedProvider(data.defaultProvider);

          // Find the default provider and set its default model
          const selectedProvider = data.providers.find((p:Provider) => p.id === data.defaultProvider);
          if (selectedProvider && selectedProvider.models.length > 0) {
            setSelectedModel(selectedProvider.models[0].id);
          }
        } else {
          setSelectedProvider(providerRef.current);
          setSelectedModel(modelRef.current);
        }
      } catch (err) {
        console.error('Failed to fetch model configurations:', err);
      } finally {
        setIsLoading(false);
      }
    };
    if(provider == '' || model == '') {
      fetchModel()
    }
  }, [provider, model]);

  const clearConversation = () => {
    setQuestion('');
    setResponse('');
    setConversationHistory([]);
    setResearchIteration(0);
    setResearchComplete(false);
    setResearchStages([]);
    setThinkingExpanded(false);
    if (inputRef.current) {
      inputRef.current.focus();
    }
  };
  const downloadresponse = () =>{
  const blob = new Blob([response], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `response-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.md`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

  // Function to check if research is complete based on response content
  const checkIfResearchComplete = (content: string): boolean => {
    // Server-side orchestration sends a [RESEARCH_EVENT] with type "complete"
    // so we just check for the final synthesis marker
    if (content.includes('## 📝 Final Synthesis')) return true;
    if (content.includes('## Final Conclusion')) return true;
    if (content.includes('"type":"complete"')) return true;
    return false;
  };

  /**
   * Strip JSON artifacts and tool-status lines that should never appear
   * in the user-visible answer.
   */
  const cleanDisplayText = (text: string): string => {
    if (!text) return text;
    // Remove [RESEARCH_EVENT]{...} markers (with nested braces) using iterative brace-counting
    let cleaned = text;
    const evtMarker = '[RESEARCH_EVENT]';
    let pos = cleaned.indexOf(evtMarker);
    while (pos !== -1) {
      const jStart = pos + evtMarker.length;
      if (jStart < cleaned.length && cleaned[jStart] === '{') {
        let depth = 0, inStr = false, esc = false, end = -1;
        for (let i = jStart; i < cleaned.length; i++) {
          const ch = cleaned[i];
          if (esc) { esc = false; continue; }
          if (ch === '\\') { esc = true; continue; }
          if (ch === '"') { inStr = !inStr; continue; }
          if (inStr) continue;
          if (ch === '{') depth++;
          else if (ch === '}') { depth--; if (depth === 0) { end = i + 1; break; } }
        }
        if (end !== -1) {
          cleaned = cleaned.slice(0, pos) + cleaned.slice(end);
        } else { break; }
      } else {
        pos = pos + evtMarker.length;
      }
      pos = cleaned.indexOf(evtMarker, pos);
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
   * displayText comes EXCLUSIVELY from the conclusion event's data field,
   * ensuring no intermediate JSON, emoji-prefixed text, or thinking
   * process ever leaks into the visible answer.
   */
  const parseResearchEvents = (raw: string): { displayText: string; stages: ResearchStage[] } => {
    const stages: ResearchStage[] = [];
    let iterationCounter = 0;
    let conclusionText = '';
    const marker = '[RESEARCH_EVENT]';
    const markerLen = marker.length;

    let searchFrom = 0;
    while (true) {
      const idx = raw.indexOf(marker, searchFrom);
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

    // displayText is ONLY the conclusion data — nothing else.
    // All intermediate text (emoji lines, JSON, findings) stays hidden in stages.
    return { displayText: conclusionText, stages };
  };

  // WebSocket reference
  const webSocketRef = useRef<WebSocket | null>(null);

  // Fallback to HTTP if WebSocket fails
  const fallbackToHttp = async (requestBody: ChatCompletionRequest) => {
    try {
      const apiResponse = await fetch(`/api/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      });

      if (!apiResponse.ok) throw new Error(`API error: ${apiResponse.status}`);

      const reader = apiResponse.body?.getReader();
      const decoder = new TextDecoder();
      if (!reader) throw new Error('Failed to get response reader');

      let fullResponse = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        fullResponse += chunk;

        if (deepResearch) {
          const { displayText, stages } = parseResearchEvents(fullResponse);
          setResponse(cleanDisplayText(displayText));
          if (stages.length > 0) {
            setResearchStages(stages);
            setResearchIteration(stages.length);
          }
        } else {
          setResponse(fullResponse);
        }
      }

      if (deepResearch) {
        setResearchComplete(true);
        setThinkingExpanded(false);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!question.trim() || isLoading) return;

    handleConfirmAsk();
  };

  // Handle confirm and send request
  const handleConfirmAsk = async () => {
    setIsLoading(true);
    setResponse('');
    setResearchIteration(0);
    setResearchComplete(false);
    setResearchStages([]);
    setThinkingExpanded(false);

    try {
      // Create initial message
      const initialMessage: Message = {
        role: 'user',
        content: deepResearch ? `[DEEP RESEARCH] ${question}` : question
      };

      // Set initial conversation history
      const newHistory: Message[] = [initialMessage];
      setConversationHistory(newHistory);

      // Prepare request body
      const requestBody: ChatCompletionRequest = {
        repo_url: getRepoUrl(repoInfo),
        type: repoInfo.type,
        messages: newHistory.map(msg => ({ role: msg.role as 'user' | 'assistant', content: msg.content })),
        provider: selectedProvider,
        model: isCustomSelectedModel ? customSelectedModel : selectedModel,
        language: language,
        user_id: userId,
      };

      // Add tokens if available
      if (repoInfo?.token) {
        requestBody.token = repoInfo.token;
      }

      // Close any existing WebSocket connection
      closeWebSocket(webSocketRef.current);

      let fullResponse = '';

      // Create a new WebSocket connection
      webSocketRef.current = createChatWebSocket(
        requestBody,
        // Message handler
        (message: string) => {
          fullResponse += message;

          // For deep research, parse structured events from the stream
          if (deepResearch) {
            const { displayText, stages } = parseResearchEvents(fullResponse);
            setResponse(cleanDisplayText(displayText));
            if (stages.length > 0) {
              setResearchStages(stages);
              setResearchIteration(stages.length);
            }
          } else {
            setResponse(cleanDisplayText(fullResponse));
          }
        },
        // Error handler
        (error: Event) => {
          console.error('WebSocket error:', error);
          setResponse(prev => prev + '\n\nError: WebSocket connection failed. Falling back to HTTP...');

          // Fallback to HTTP if WebSocket fails
          fallbackToHttp(requestBody);
        },
        // Close handler
        () => {
          if (deepResearch) {
            // Server-side orchestration is complete when the connection closes
            setResearchComplete(true);
            setThinkingExpanded(false); // Auto-collapse thinking section
          }

          setIsLoading(false);
        }
      );
    } catch (error) {
      console.error('Error during API call:', error);
      setResponse(prev => prev + '\n\nError: Failed to get a response. Please try again.');
      setResearchComplete(true);
      setIsLoading(false);
    }
  };

  const [buttonWidth, setButtonWidth] = useState(0);
  const buttonRef = useRef<HTMLButtonElement>(null);

  // Measure button width and update state
  useEffect(() => {
    if (buttonRef.current) {
      const width = buttonRef.current.offsetWidth;
      setButtonWidth(width);
    }
  }, [messages.ask?.askButton, isLoading]);

  return (
    <div>
      <div className="p-4">
        <div className="flex items-center justify-end mb-4">
          {/* Model selection button */}
          <button
            type="button"
            onClick={() => setIsModelSelectionModalOpen(true)}
            className="text-xs px-2.5 py-1 rounded border border-[var(--border-color)]/40 bg-[var(--background)]/10 text-[var(--foreground)]/80 hover:bg-[var(--background)]/30 hover:text-[var(--foreground)] transition-colors flex items-center gap-1.5"
          >
            <span>{selectedProvider}/{isCustomSelectedModel ? customSelectedModel : selectedModel}</span>
            <svg className="h-3.5 w-3.5 text-[var(--accent-primary)]/70" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
            </svg>
          </button>
        </div>

        {/* Question input */}
        <form onSubmit={handleSubmit} className="mt-4">
          <div className="relative">
            <input
              ref={inputRef}
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder={messages.ask?.placeholder || 'What would you like to know about this codebase?'}
              className="block w-full rounded-md border border-[var(--border-color)] bg-[var(--input-bg)] text-[var(--foreground)] px-5 py-3.5 text-base shadow-sm focus:border-[var(--accent-primary)] focus:ring-2 focus:ring-[var(--accent-primary)]/30 focus:outline-none transition-all"
              style={{ paddingRight: `${buttonWidth + 24}px` }}
              disabled={isLoading}
            />
            <button
              ref={buttonRef}
              type="submit"
              disabled={isLoading || !question.trim()}
              className={`absolute right-3 top-1/2 transform -translate-y-1/2 px-4 py-2 rounded-md font-medium text-sm ${
                isLoading || !question.trim()
                  ? 'bg-[var(--button-disabled-bg)] text-[var(--button-disabled-text)] cursor-not-allowed'
                  : 'bg-[var(--accent-primary)] text-white hover:bg-[var(--accent-primary)]/90 shadow-sm'
              } transition-all duration-200 flex items-center gap-1.5`}
            >
              {isLoading ? (
                <div className="w-4 h-4 rounded-full border-2 border-t-transparent border-white animate-spin" />
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                  </svg>
                  <span>{messages.ask?.askButton || 'Ask'}</span>
                </>
              )}
            </button>
          </div>

          {/* Deep Research toggle */}
          <div className="flex items-center mt-2 justify-between">
            <div className="group relative">
              <label className="flex items-center cursor-pointer">
                <span className="text-xs text-gray-600 dark:text-gray-400 mr-2">Deep Research</span>
                <div className="relative">
                  <input
                    type="checkbox"
                    checked={deepResearch}
                    onChange={() => setDeepResearch(!deepResearch)}
                    className="sr-only"
                  />
                  <div className={`w-10 h-5 rounded-full transition-colors ${deepResearch ? 'bg-teal-600' : 'bg-gray-300 dark:bg-gray-600'}`}></div>
                  <div className={`absolute left-0.5 top-0.5 w-4 h-4 rounded-full bg-white transition-transform transform ${deepResearch ? 'translate-x-5' : ''}`}></div>
                </div>
              </label>
              <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block bg-gray-800 text-white text-xs rounded p-2 w-72 z-10">
                <div className="relative">
                  <div className="absolute -bottom-2 left-4 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-800"></div>
                  <p className="mb-1">Deep Research conducts a multi-turn investigation process:</p>
                  <ul className="list-disc pl-4 text-xs">
                    <li><strong>Initial Research:</strong> Creates a research plan and initial findings</li>
                    <li><strong>Iteration 1:</strong> Explores specific aspects in depth</li>
                    <li><strong>Iteration 2:</strong> Investigates remaining questions</li>
                    <li><strong>Iterations 3-4:</strong> Dives deeper into complex areas</li>
                    <li><strong>Final Conclusion:</strong> Comprehensive answer based on all iterations</li>
                  </ul>
                  <p className="mt-1 text-xs italic">The AI automatically continues research until complete (up to 5 iterations)</p>
                </div>
              </div>
            </div>
            {deepResearch && (
              <div className="text-xs text-teal-600 dark:text-teal-400">
                Multi-turn research process enabled
                {researchIteration > 0 && !researchComplete && ` (iteration ${researchIteration})`}
                {researchComplete && ` (complete)`}
              </div>
            )}
          </div>
        </form>

        {/* Response area */}
        {(response || (deepResearch && researchStages.length > 0)) && (
          <div className="border-t border-gray-200 dark:border-gray-700 mt-4">
            <div
              ref={responseRef}
              className="p-4 max-h-[500px] overflow-y-auto"
            >
              {/* Inline collapsible thinking section */}
              {deepResearch && researchStages.length > 0 && (
                <div className="mb-3">
                  <button
                    onClick={() => setThinkingExpanded(!thinkingExpanded)}
                    className="flex items-center gap-1.5 text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
                  >
                    <svg
                      className={`w-3 h-3 transition-transform ${thinkingExpanded ? 'rotate-90' : ''}`}
                      fill="none" viewBox="0 0 24 24" stroke="currentColor"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                    <span>🔬</span>
                    <span className="font-medium">Research Process</span>
                    <span className="text-gray-400 dark:text-gray-500">
                      ({researchStages.filter(s => s.type === 'update').length} iterations)
                    </span>
                    {isLoading && !researchComplete && (
                      <span className="inline-block w-1.5 h-1.5 bg-teal-500 rounded-full animate-pulse" />
                    )}
                  </button>
                  {thinkingExpanded && (
                    <div className="mt-1.5 pl-4 border-l-2 border-gray-200 dark:border-gray-700 max-h-[250px] overflow-y-auto">
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
                              <span className="text-gray-700 dark:text-gray-300">{stage.title}</span>
                              {stage.content && stage.type !== 'tool_call' && (
                                <p className="text-gray-500 dark:text-gray-500 mt-0.5 line-clamp-2">{stage.content.slice(0, 150)}</p>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
              {response && <Markdown content={response} />}
            </div>

            {/* Action buttons */}
            <div className="p-2 flex justify-end items-center border-t border-gray-200 dark:border-gray-700">

            <div className="flex items-center space-x-2">
              {/* Download button */}
              <button
                onClick={downloadresponse}
                className="text-xs text-gray-500 dark:text-gray-400 hover:text-green-600 dark:hover:text-green-400 px-2 py-1 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 flex items-center gap-1"
                title="Download response as markdown file"
              >
                <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Download
              </button>

              {/* Clear button */}
              <button
                id="ask-clear-conversation"
                onClick={clearConversation}
                className="text-xs text-gray-500 dark:text-gray-400 hover:text-teal-600 dark:hover:text-teal-400 px-2 py-1 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700"
              >
                Clear conversation
              </button>
            </div>
              </div>
          </div>
        )}

        {/* Loading indicator */}
        {isLoading && !response && (
          <div className="p-4 border-t border-gray-200 dark:border-gray-700">
            <div className="flex items-center space-x-2">
              <div className="animate-pulse flex space-x-1">
                <div className="h-2 w-2 bg-teal-600 rounded-full"></div>
                <div className="h-2 w-2 bg-teal-600 rounded-full"></div>
                <div className="h-2 w-2 bg-teal-600 rounded-full"></div>
              </div>
              <span className="text-xs text-gray-500 dark:text-gray-400">
                {deepResearch
                  ? (researchIteration === 0
                    ? "Planning research approach..."
                    : `Research iteration ${researchIteration} in progress...`)
                  : "Thinking..."}
              </span>
            </div>
            {deepResearch && (
              <div className="mt-2 text-xs text-gray-500 dark:text-gray-400 pl-5">
                <div className="flex flex-col space-y-1">
                  {researchIteration === 0 && (
                    <>
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-blue-500 rounded-full mr-2"></div>
                        <span>Creating research plan...</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                        <span>Identifying key areas to investigate...</span>
                      </div>
                    </>
                  )}
                  {researchIteration === 1 && (
                    <>
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-blue-500 rounded-full mr-2"></div>
                        <span>Exploring first research area in depth...</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                        <span>Analyzing code patterns and structures...</span>
                      </div>
                    </>
                  )}
                  {researchIteration === 2 && (
                    <>
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-amber-500 rounded-full mr-2"></div>
                        <span>Investigating remaining questions...</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-teal-500 rounded-full mr-2"></div>
                        <span>Connecting findings from previous iterations...</span>
                      </div>
                    </>
                  )}
                  {researchIteration === 3 && (
                    <>
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-indigo-500 rounded-full mr-2"></div>
                        <span>Exploring deeper connections...</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-blue-500 rounded-full mr-2"></div>
                        <span>Analyzing complex patterns...</span>
                      </div>
                    </>
                  )}
                  {researchIteration === 4 && (
                    <>
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-teal-500 rounded-full mr-2"></div>
                        <span>Refining research conclusions...</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-cyan-500 rounded-full mr-2"></div>
                        <span>Addressing remaining edge cases...</span>
                      </div>
                    </>
                  )}
                  {researchIteration >= 5 && (
                    <>
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-teal-500 rounded-full mr-2"></div>
                        <span>Finalizing comprehensive answer...</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                        <span>Synthesizing all research findings...</span>
                      </div>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

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
    </div>
  );
};

export default Ask;
