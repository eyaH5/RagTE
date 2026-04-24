import { useState, useRef, useEffect, useCallback } from 'react';
import { queryApi } from './queryApi';
import type { ChatMessage } from '../../types';

let msgCounter = 0;

export function useChat(options?: { sourceFilter?: string[]; universeId?: string }) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const send = useCallback(async (question: string) => {
    if (!question.trim() || loading) return;

    const userMsg: ChatMessage = {
      id: `msg-${++msgCounter}`,
      role: 'user',
      content: question,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const res = await queryApi.ask(
        question,
        6,
        options?.sourceFilter,
        options?.universeId,
      );
      const data = res.data;
      const assistantMsg: ChatMessage = {
        id: `msg-${++msgCounter}`,
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
        query_time_ms: data.query_time_ms,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: any) {
      const errorMsg: ChatMessage = {
        id: `msg-${++msgCounter}`,
        role: 'assistant',
        content: `⚠️ Erreur: ${err.response?.data?.detail || err.message}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }, [loading, options?.sourceFilter, options?.universeId]);

  const clear = useCallback(() => {
    setMessages([]);
  }, []);

  return {
    messages,
    loading,
    bottomRef,
    inputRef,
    send,
    clear,
  };
}
