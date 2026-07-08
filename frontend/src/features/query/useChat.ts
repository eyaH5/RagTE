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
    const assistantId = `msg-${++msgCounter}`;
    const assistantMsg: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      sources: [],
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setLoading(true);

    try {
      const res = await queryApi.stream(
        question,
        options?.sourceFilter?.length === 1 ? 2 : 3,
        options?.sourceFilter,
        options?.universeId,
      );

      if (!res.ok || !res.body) {
        throw new Error((await res.text()) || `HTTP ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        while (buffer.includes('\n\n')) {
          const idx = buffer.indexOf('\n\n');
          const rawEvent = buffer.slice(0, idx);
          buffer = buffer.slice(idx + 2);

          const lines = rawEvent.split('\n');
          let eventType = 'message';
          let data = '';

          for (const line of lines) {
            if (line.startsWith('event:')) eventType = line.slice(6).trim();
            if (line.startsWith('data:')) data += line.slice(5).trim();
          }

          if (!data) continue;
          const payload = JSON.parse(data);

          if (eventType === 'meta') {
            setMessages((prev) =>
              prev.map((msg) => (msg.id === assistantId ? { ...msg, sources: payload.sources || [] } : msg)),
            );
          } else if (eventType === 'token') {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId
                  ? { ...msg, content: `${msg.content}${payload.text || ''}` }
                  : msg,
              ),
            );
          } else if (eventType === 'done') {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId
                  ? {
                      ...msg,
                      content: payload.answer || msg.content || 'Aucune reponse exploitable n a ete retournee.',
                      query_time_ms: payload.query_time_ms,
                    }
                  : msg,
              ),
            );
          }
        }
      }
    } catch (err: any) {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantId
            ? { ...msg, content: `Erreur: ${err.response?.data?.detail || err.message}` }
            : msg,
        ),
      );
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
