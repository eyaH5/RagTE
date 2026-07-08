import { useState, type FormEvent } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useChat } from '.';
import { useAuth } from '../../AuthContext';
import type { SourceCitation } from '../../types';
import { Send, FileText, Clock, Zap, Sparkles, X } from 'lucide-react';
import './Chat.css';

const CHAT_SUGGESTIONS = [
  "Quel est l'objet du cahier des charges / de la présente consultation ?",
  "Quel est le mode d'envoi ou de dépôt de la soumission ?",
  "Quelle est la date limite réelle de soumission ?",
  "Quelle est la durée de validité de l'offre ?",
  "Quelle est la date / modalité d'ouverture des plis ?",
  "Quel est le montant / l'exigence de la caution provisoire ?",
  "Quels sont les documents administratifs exigés ?",
  "Quelle documentation technique est exigée ?",
  "Quels sont les documents financiers exigés ?",
  "Quelle est la période de garantie exigée ?",
  "Existe-t-il des pénalités de retard ?",
  "Quelles sont les modalités de paiement ?",
];

export default function ChatPage() {
  const { user } = useAuth();
  const [searchParams, setSearchParams] = useSearchParams();
  const [input, setInput] = useState('');

  const activeSource = searchParams.get('source');
  const sourceFilter = activeSource ? [activeSource] : undefined;

  const { messages, loading, bottomRef, inputRef, send } = useChat({ sourceFilter });

  const clearSource = () => {
    searchParams.delete('source');
    setSearchParams(searchParams);
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    const question = input.trim();
    if (!question || loading) return;
    setInput('');
    await send(question);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="chat-page">
      <div
        className="chat-header"
        style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
      >
        <div>
          <h1>
            <Sparkles size={22} style={{ verticalAlign: 'middle', marginRight: 8 }} />
            Chat
          </h1>
          <p className="text-secondary text-sm">
            Interrogez vos documents - departement {user?.department_id}
          </p>
        </div>
        {activeSource && (
          <div className="chat-header__meta">
            <span
              className="badge badge-info"
              style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}
            >
              <FileText size={12} />
              Cible: {activeSource}
              <button
                onClick={clearSource}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'inherit',
                  cursor: 'pointer',
                  padding: '0 2px',
                }}
              >
                <X size={12} />
              </button>
            </span>
          </div>
        )}
      </div>

      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty animate-fade-in">
            <div className="chat-empty__icon">
              <Sparkles size={48} />
            </div>
            <h2>Pret a discuter</h2>
            <p className="text-secondary">
              Posez une question sur vos documents.
              <br />
              Le systeme cherchera dans vos documents autorises.
            </p>
            <div className="chat-suggestions">
              {CHAT_SUGGESTIONS.map((q) => (
                <button
                  key={q}
                  className="chat-suggestion"
                  onClick={() => {
                    setInput(q);
                    inputRef.current?.focus();
                  }}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`chat-msg chat-msg--${msg.role} animate-fade-in`}
          >
            <div className="chat-msg__content">
              <div className="chat-msg__text">
                {msg.role === 'assistant' && loading && !msg.content ? (
                  <div className="chat-thinking">
                    <div className="spinner" />
                    <span>Recherche en cours...</span>
                  </div>
                ) : (
                  msg.content
                )}
              </div>
              {msg.sources && msg.sources.length > 0 && (
                <SourceList sources={msg.sources} />
              )}
              {msg.query_time_ms !== undefined && (
                <div className="chat-msg__meta">
                  <Clock size={12} />
                  <span>{msg.query_time_ms}ms</span>
                  <Zap size={12} />
                  <span>{msg.sources?.length || 0} sources</span>
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <form className="chat-input-bar" onSubmit={handleSubmit}>
        <textarea
          ref={inputRef}
          className="chat-input"
          placeholder="Posez votre question sur les documents..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
          disabled={loading}
        />
        <button
          type="submit"
          className="btn btn-primary chat-send"
          disabled={!input.trim() || loading}
        >
          <Send size={18} />
        </button>
      </form>
    </div>
  );
}

function SourceList({ sources }: { sources: SourceCitation[] }) {
  return (
    <div className="chat-sources">
      <div className="chat-sources__label">
        <FileText size={12} /> Sources
      </div>
      <div className="chat-sources__list">
        {sources.map((s, i) => (
          <div key={i} className="chat-source">
            <span className="chat-source__name">{s.source}</span>
            <span className="chat-source__page">p.{s.page}</span>
            {s.score > 0 && (
              <span className="chat-source__score">{(s.score * 100).toFixed(0)}%</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
