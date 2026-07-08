import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Upload, FileText, Trash2, Eye, CheckCircle, AlertCircle,
  Loader, RefreshCw, Clock, MessageSquare, Search, X
} from 'lucide-react';
import { useAuth } from '../../AuthContext';
import type { Document } from '../../types';
import { useDocuments } from '.';
import './Documents.css';

const STATUS_MAP: Record<string, { badge: string; icon: typeof CheckCircle; label: string }> = {
  queued: { badge: 'badge-processing', icon: Clock, label: 'En attente' },
  indexed: { badge: 'badge-success', icon: CheckCircle, label: 'Indexe' },
  processing: { badge: 'badge-processing', icon: Loader, label: 'En cours' },
  failed: { badge: 'badge-error', icon: AlertCircle, label: 'Echec' },
};

const VISIBILITY_OPTIONS = [
  { value: 'private', label: 'Prive' },
  { value: 'department', label: 'Departement' },
] as const;

type UploadVisibility = (typeof VISIBILITY_OPTIONS)[number]['value'];

const isActiveStatus = (status: Document['status']) => status === 'queued' || status === 'processing';

function progressForDocument(doc: Document) {
  if (doc.status === 'indexed') return 100;
  if (doc.status === 'failed') return 100;
  if (doc.status === 'processing') return doc.chunk_count > 0 ? 88 : 70;
  return 45;
}

function uploadProgressLabel(phase: 'uploading' | 'queued' | 'processing' | 'complete' | 'failed') {
  if (phase === 'uploading') return 'Televersement';
  if (phase === 'queued') return 'En file';
  if (phase === 'processing') return 'Indexation';
  if (phase === 'complete') return 'Termine';
  return 'Echec';
}

export default function DocumentsPage() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const {
    documents, loading, uploading, uploadMsg, uploadProgress, fileRef,
    fetchDocs, upload, remove, analyze, updateVisibility,
  } = useDocuments();

  const [analysisModalOpen, setAnalysisModalOpen] = useState(false);
  const [analysisDoc, setAnalysisDoc] = useState<Document | null>(null);
  const [analysisType, setAnalysisType] = useState('tender_checklist');
  const [analysisResult, setAnalysisResult] = useState<string | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [uploadVisibility, setUploadVisibility] = useState<UploadVisibility>('department');

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    await upload(file, uploadVisibility);
  };

  const handleDelete = async (doc: Document) => {
    if (!confirm(`Supprimer "${doc.filename}" ? Cette action est irreversible.`)) return;
    const ok = await remove(doc);
    if (!ok) alert('Erreur lors de la suppression.');
  };

  const runAnalysis = async (doc: Document, type = 'tender_checklist') => {
    setAnalysisLoading(true);
    try {
      const result = await analyze(doc.id, type);
      setAnalysisResult(result);
    } catch {
      setAnalysisResult("Erreur lors de l'analyse du document.");
    } finally {
      setAnalysisLoading(false);
    }
  };

  const handleAnalyze = async () => {
    if (!analysisDoc) return;
    await runAnalysis(analysisDoc, analysisType);
  };

  const openAnalysisModal = (doc: Document) => {
    setAnalysisDoc(doc);
    setAnalysisType('tender_checklist');
    setAnalysisResult(null);
    setAnalysisModalOpen(true);
    void runAnalysis(doc, 'tender_checklist');
  };

  const handleChat = (filename: string) => {
    navigate(`/chat?source=${encodeURIComponent(filename)}`);
  };

  const handleVisibility = async (doc: Document, visibility: UploadVisibility) => {
    try {
      await updateVisibility(doc, visibility);
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Erreur');
    }
  };

  return (
    <div className="docs-page">
      <div className="docs-header">
        <div>
          <h1>
            <FileText size={22} style={{ verticalAlign: 'middle', marginRight: 8 }} />
            Documents
          </h1>
          <p className="text-secondary text-sm">
            {documents.length} document(s) - departement {user?.department_id}
          </p>
        </div>
        <div className="docs-actions">
          <button className="btn btn-ghost" onClick={() => void fetchDocs()} disabled={loading}>
            <RefreshCw size={16} className={loading ? 'spin-icon' : ''} />
            Actualiser
          </button>
          <label className="text-sm text-secondary docs-visibility-control">
            Visibilite
            <select
              className="doc-card__select"
              value={uploadVisibility}
              onChange={(e) => setUploadVisibility(e.target.value as UploadVisibility)}
              disabled={uploading}
            >
              {VISIBILITY_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </label>
          <label className={`btn btn-primary ${uploading ? 'btn--loading' : ''}`}>
            {uploading ? <div className="spinner" /> : <Upload size={16} />}
            {uploading ? 'Import...' : 'Importer fichier'}
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
      </div>

      {uploadMsg && (
        <div className={`docs-alert docs-alert--${uploadMsg.type} animate-fade-in`}>
          {uploadMsg.type === 'success' ? <CheckCircle size={16} /> : <AlertCircle size={16} />}
          {uploadMsg.text}
        </div>
      )}

      {uploadProgress && (
        <div className={`docs-upload-progress docs-upload-progress--${uploadProgress.phase} animate-fade-in`}>
          <div className="docs-upload-progress__top">
            <div className="docs-upload-progress__title">
              {uploadProgress.phase === 'failed' ? (
                <AlertCircle size={16} />
              ) : uploadProgress.phase === 'complete' ? (
                <CheckCircle size={16} />
              ) : (
                <Loader size={16} className="spin-icon" />
              )}
              <span>{uploadProgressLabel(uploadProgress.phase)}</span>
            </div>
            <span className="docs-upload-progress__percent">{uploadProgress.percent}%</span>
          </div>
          <div className="docs-upload-progress__filename" title={uploadProgress.filename}>
            {uploadProgress.filename}
          </div>
          <div className="progress-bar" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={uploadProgress.percent}>
            <div
              className={`progress-bar__fill ${uploadProgress.phase === 'queued' || uploadProgress.phase === 'processing' ? 'progress-bar__fill--animated' : ''}`}
              style={{ width: `${uploadProgress.percent}%` }}
            />
          </div>
        </div>
      )}

      {loading ? (
        <div className="docs-loading">
          <div className="spinner spinner-lg" />
          <span>Chargement...</span>
        </div>
      ) : documents.length === 0 ? (
        <div className="docs-empty animate-fade-in">
          <FileText size={48} color="var(--text-muted)" />
          <h3>Aucun document</h3>
          <p className="text-secondary">Importez un fichier pour commencer.</p>
        </div>
      ) : (
        <div className="docs-grid">
          {documents.map((doc) => {
            const status = STATUS_MAP[doc.status] || STATUS_MAP.processing;
            const StatusIcon = status.icon;
            const cardProgress = progressForDocument(doc);
            return (
              <div key={doc.id} className="doc-card card animate-fade-in">
                <div className="doc-card__header">
                  <FileText size={20} color="var(--brand-primary-light)" />
                  <span className="doc-card__name" title={doc.filename}>
                    {doc.filename}
                  </span>
                </div>

                <div className="doc-card__meta">
                  <span className={`badge ${status.badge}`}>
                    <StatusIcon size={10} className={doc.status === 'processing' ? 'spin-icon' : ''} />
                    {status.label}
                  </span>
                  <span className="text-xs text-muted">
                    {doc.chunk_count} chunks
                  </span>
                </div>

                {isActiveStatus(doc.status) && (
                  <div className="doc-card__progress">
                    <div className="doc-card__progress-row">
                      <span>{doc.status === 'queued' ? 'En file d attente' : 'Indexation en cours'}</span>
                      <span>{cardProgress}%</span>
                    </div>
                    <div className="progress-bar progress-bar--compact" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={cardProgress}>
                      <div
                        className="progress-bar__fill progress-bar__fill--animated"
                        style={{ width: `${cardProgress}%` }}
                      />
                    </div>
                  </div>
                )}

                <div className="doc-card__details">
                  <div className="doc-card__detail">
                    <Clock size={12} />
                    <span>{new Date(doc.created_at).toLocaleDateString('fr-FR')}</span>
                  </div>
                  <div className="doc-card__detail">
                    <Eye size={12} />
                    <select
                      className="doc-card__select"
                      value={doc.visibility}
                      onChange={(e) => handleVisibility(doc, e.target.value as UploadVisibility)}
                      disabled={user?.role !== 'admin' && doc.uploaded_by !== user?.id}
                    >
                      {VISIBILITY_OPTIONS.map((opt) => (
                        <option key={opt.value} value={opt.value}>{opt.label}</option>
                      ))}
                    </select>
                  </div>
                </div>

                <div className="doc-card__actions flex gap-sm" style={{ marginTop: 'var(--space-md)' }}>
                  <button
                    className="btn btn-primary btn-sm w-full"
                    onClick={() => handleChat(doc.filename)}
                    disabled={doc.status !== 'indexed'}
                  >
                    <MessageSquare size={14} />
                    Discuter
                  </button>
                  <button
                    className="btn btn-ghost btn-sm w-full"
                    onClick={() => openAnalysisModal(doc)}
                    disabled={doc.status !== 'indexed'}
                  >
                    <Search size={14} />
                    Analyser
                  </button>
                </div>

                {user?.role === 'admin' && (
                  <button
                    className="btn btn-ghost btn-sm doc-card__delete mt-sm w-full"
                    style={{ color: 'var(--status-error)' }}
                    onClick={() => handleDelete(doc)}
                  >
                    <Trash2 size={14} />
                    Supprimer
                  </button>
                )}
              </div>
            );
          })}
        </div>
      )}

      {analysisModalOpen && analysisDoc && (
        <div className="modal-overlay" onClick={() => setAnalysisModalOpen(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '800px', width: '90%', maxHeight: '90vh', display: 'flex', flexDirection: 'column' }}>
            <div className="modal-header flex justify-between items-center mb-md">
              <h3>Analyse: {analysisDoc.filename}</h3>
              <button className="btn btn-icon btn-ghost" onClick={() => setAnalysisModalOpen(false)}>
                <X size={20} />
              </button>
            </div>

            <div className="flex gap-md mb-md justify-end">
              <button
                className="btn btn-primary"
                onClick={handleAnalyze}
                disabled={analysisLoading}
                style={{ whiteSpace: 'nowrap' }}
              >
                {analysisLoading ? <div className="spinner" style={{ width: '16px', height: '16px', borderWidth: '2px' }} /> : <Search size={16} />}
                Relancer l'analyse
              </button>
            </div>

            <div className="analysis-result-container card flex-col" style={{ flexGrow: 1, overflowY: 'auto', background: 'var(--bg-input)' }}>
              {analysisLoading ? (
                <div className="flex-col items-center justify-center gap-md py-lg" style={{ color: 'var(--text-muted)' }}>
                  <div className="spinner spinner-lg" />
                  <p>Analyse en cours...</p>
                </div>
              ) : analysisResult ? (
                <div className="markdown-body" style={{ whiteSpace: 'pre-wrap', lineHeight: '1.6' }}>
                  {analysisResult}
                </div>
              ) : (
                <div className="flex items-center justify-center" style={{ height: '100%', color: 'var(--text-muted)' }}>
                  Analyse non lancee.
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
