import { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useUniverseView } from '.';
import { useChat } from '../query';
import { useAuth } from '../../AuthContext';
import {
  ArrowLeft, FileText, Upload, Send, Brain,
  CheckCircle, Loader, AlertCircle, Trash2,
} from 'lucide-react';
import './UniverseView.css';

const UPLOAD_VISIBILITY_OPTIONS = [
  { value: 'private', label: 'Seulement moi' },
  { value: 'department', label: 'Mon département' },
] as const;

type UploadVisibility = (typeof UPLOAD_VISIBILITY_OPTIONS)[number]['value'];

const DEPT_COLORS: Record<string, string> = {
  backoffice: '#3b82f6',
  software: '#10b981',
  commerciale: '#f59e0b',
  infrastructure: '#8b5cf6',
  admin: '#6b7280',
};

const DEPT_LABELS: Record<string, string> = {
  backoffice: 'Agent Back Office',
  software: 'Développeur Logiciel',
  commerciale: 'Analyste Commercial',
  infrastructure: 'Ingénieur Infrastructure',
  admin: 'Administrateur',
};

const STATUS_ICON: Record<string, typeof CheckCircle> = {
  indexed: CheckCircle,
  processing: Loader,
  failed: AlertCircle,
};

export default function UniverseViewPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { user } = useAuth();

  const {
    universe, documents, loading, uploading, fileRef,
    upload, deleteUniverse,
  } = useUniverseView(id);

  const { messages, loading: asking, bottomRef, send } = useChat({ universeId: id });
  const [input, setInput] = useState('');
  const [uploadVisibility, setUploadVisibility] = useState<UploadVisibility>('department');

  const handleSend = async () => {
    if (!input.trim() || asking) return;
    const question = input.trim();
    setInput('');
    await send(question);
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      await upload(file, uploadVisibility);
    } catch (err: any) {
      alert(err.response?.data?.detail || "Erreur lors de l'import");
    }
  };

  const handleDelete = async () => {
    if (!universe) return;
    if (!confirm(`Supprimer l'universe "${universe.name}" ?\n\nTous les documents et conversations associés seront supprimés.\nCette action est irréversible.`)) return;
    try {
      await deleteUniverse();
      navigate('/universes');
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Erreur lors de la suppression');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  if (loading) {
    return (
      <div className="universe-view" style={{ alignItems: 'center', justifyContent: 'center', display: 'flex' }}>
        <div className="spinner spinner-lg" />
      </div>
    );
  }

  if (!universe) {
    navigate('/universes');
    return null;
  }

  const deptColor = DEPT_COLORS[universe.department_id] || 'var(--brand-primary)';
  const personaLabel = DEPT_LABELS[universe.department_id] || 'Assistant IA';
  const canUpload = user && ['admin', 'manager', 'analyst'].includes(user.role);

  return (
    <div className="universe-view">
      {/* ── Top Bar ──────────────────────────────────────── */}
      <div className="universe-topbar">
        <button className="universe-topbar__back" onClick={() => navigate('/universes')}>
          <ArrowLeft size={16} />
          Universes
        </button>

        <div className="universe-topbar__info">
          <div className="universe-topbar__name">{universe.name}</div>
          <div className="universe-topbar__persona">
            <span
              className="universe-topbar__persona-dot"
              style={{ background: deptColor }}
            />
            🧠 {personaLabel} AI
          </div>
        </div>

        {user?.role === 'admin' && (
          <button
            className="btn btn-ghost btn-sm"
            onClick={handleDelete}
            title="Supprimer l'universe"
            style={{ color: 'var(--status-error)', borderColor: 'transparent' }}
          >
            <Trash2 size={16} />
          </button>
        )}
      </div>

      {/* ── Split Workspace ──────────────────────────────── */}
      <div className="universe-workspace">
        {/* Left: Knowledge Base */}
        <div className="universe-docs">
          <div className="universe-docs__header">
            <h3>
              <FileText size={16} />
              Base de connaissances
            </h3>
            <span className="text-xs text-muted">{documents.length} doc(s)</span>
          </div>

          <div className="universe-docs__list">
            {documents.length === 0 ? (
              <div className="universe-docs__empty">
                <FileText size={32} color="var(--text-muted)" />
                <p>Aucun document.<br />Importez des fichiers pour commencer.</p>
              </div>
            ) : (
              documents.map((doc) => {
                const Icon = STATUS_ICON[doc.status] || FileText;
                const statusColor = doc.status === 'indexed'
                  ? 'var(--status-success)'
                  : doc.status === 'failed'
                    ? 'var(--status-error)'
                    : 'var(--status-processing)';
                return (
                  <div key={doc.id} className="universe-doc-item">
                    <Icon size={14} color={statusColor} />
                    <span className="universe-doc-item__name" title={doc.filename}>
                      {doc.filename}
                    </span>
                    <span className="universe-doc-item__chunks">
                      {doc.chunk_count} chunks
                    </span>
                  </div>
                );
              })
            )}
          </div>

          {/* Upload zone */}
          {canUpload && (
            <div className="universe-upload">
              <div style={{ marginBottom: 'var(--space-sm)' }}>
                <select
                  className="input-field"
                  value={uploadVisibility}
                  onChange={(e) => setUploadVisibility(e.target.value as UploadVisibility)}
                  disabled={uploading}
                >
                  {UPLOAD_VISIBILITY_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              </div>
              <label className={`universe-upload__zone ${uploading ? 'universe-upload__zone--active' : ''}`}>
                {uploading ? (
                  <>
                    <div className="spinner" style={{ margin: '0 auto var(--space-xs)' }} />
                    Import en cours…
                  </>
                ) : (
                  <>
                    <Upload size={18} style={{ marginBottom: 4 }} />
                    <br />
                    Glisser un fichier ou cliquer
                  </>
                )}
                <input
                  ref={fileRef}
                  type="file"
                  accept=".pdf,.docx,.txt,.md,.csv,.json,.xlsx"
                  onChange={handleUpload}
                  hidden
                  disabled={uploading}
                />
              </label>
            </div>
          )}
        </div>

        {/* Right: Chat */}
        <div className="universe-chat">
          <div className="universe-chat__messages">
            {messages.length === 0 ? (
              <div className="universe-chat__empty">
                <Brain size={48} color="var(--text-muted)" />
                <h3>Posez une question à l'IA</h3>
                <p>
                  L'IA ({personaLabel}) analysera les {documents.length} document(s)
                  de cet universe pour vous répondre.
                </p>
              </div>
            ) : (
              messages.map((msg) => (
                <div key={msg.id} className={`chat-msg chat-msg--${msg.role}`}>
                  <div>{msg.content}</div>
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="chat-msg__sources">
                      📎 {msg.sources.map((s) => `${s.source} (p.${s.page})`).join(' · ')}
                      {msg.query_time_ms && ` · ${msg.query_time_ms}ms`}
                    </div>
                  )}
                </div>
              ))
            )}
            {asking && (
              <div className="chat-typing">
                <div className="spinner" />
                L'IA réfléchit…
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <div className="universe-chat__input-area">
            <div className="universe-chat__input-row">
              <input
                className="input-field"
                placeholder="Posez votre question..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={asking}
              />
              <button
                className="btn btn-primary"
                onClick={handleSend}
                disabled={asking || !input.trim()}
              >
                <Send size={16} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
