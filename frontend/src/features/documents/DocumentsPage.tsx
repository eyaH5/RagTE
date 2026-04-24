import { useState } from 'react';
import { useDocuments } from '.';
import { useAuth } from '../../AuthContext';
import type { Document } from '../../types';
import { useNavigate } from 'react-router-dom';
import {
  Upload, FileText, Trash2, Eye, CheckCircle, AlertCircle,
  Loader, RefreshCw, Clock, MessageSquare, Search, X
} from 'lucide-react';
import './Documents.css';

const STATUS_MAP: Record<string, { badge: string; icon: typeof CheckCircle; label: string }> = {
  indexed:    { badge: 'badge-success',    icon: CheckCircle, label: 'Indexé' },
  processing: { badge: 'badge-processing', icon: Loader,      label: 'En cours' },
  failed:     { badge: 'badge-error',      icon: AlertCircle, label: 'Échec' },
};

const VISIBILITY_OPTIONS = [
  { value: 'private', label: 'Privé' },
  { value: 'department', label: 'Département' },
  { value: 'shared', label: 'Partagé' },
];

export default function DocumentsPage() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const {
    documents, loading, uploading, uploadMsg, fileRef,
    fetchDocs, upload, remove, analyze, updateVisibility,
  } = useDocuments();

  // Analysis modal state (UI-only, stays in the page)
  const [analysisModalOpen, setAnalysisModalOpen] = useState(false);
  const [analysisDoc, setAnalysisDoc] = useState<Document | null>(null);
  const [analysisType, setAnalysisType] = useState('summary');
  const [analysisResult, setAnalysisResult] = useState<string | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    await upload(file);
  };

  const handleDelete = async (doc: Document) => {
    if (!confirm(`Supprimer "${doc.filename}" ? Cette action est irréversible.`)) return;
    const ok = await remove(doc);
    if (!ok) alert('Erreur lors de la suppression.');
  };

  const handleAnalyze = async () => {
    if (!analysisDoc) return;
    setAnalysisLoading(true);
    try {
      const result = await analyze(analysisDoc.id, analysisType);
      setAnalysisResult(result);
    } catch {
      setAnalysisResult("⚠️ Erreur lors de l'analyse du document.");
    } finally {
      setAnalysisLoading(false);
    }
  };

  const openAnalysisModal = (doc: Document) => {
    setAnalysisDoc(doc);
    setAnalysisType('summary');
    setAnalysisResult(null);
    setAnalysisModalOpen(true);
  };

  const handleChat = (filename: string) => {
    navigate(`/chat?source=${encodeURIComponent(filename)}`);
  };

  const handleVisibility = async (doc: Document, visibility: string) => {
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
            {documents.length} document(s) · département {user?.department_id}
          </p>
        </div>
        <div className="docs-actions">
          <button className="btn btn-ghost" onClick={fetchDocs} disabled={loading}>
            <RefreshCw size={16} className={loading ? 'spin-icon' : ''} />
            Actualiser
          </button>
          <label className={`btn btn-primary ${uploading ? 'btn--loading' : ''}`}>
            {uploading ? <div className="spinner" /> : <Upload size={16} />}
            {uploading ? 'Import...' : 'Importer PDF'}
            <input
              ref={fileRef}
              type="file"
              accept=".pdf"
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

      {loading ? (
        <div className="docs-loading">
          <div className="spinner spinner-lg" />
          <span>Chargement…</span>
        </div>
      ) : documents.length === 0 ? (
        <div className="docs-empty animate-fade-in">
          <FileText size={48} color="var(--text-muted)" />
          <h3>Aucun document</h3>
          <p className="text-secondary">Importez un document PDF pour commencer.</p>
        </div>
      ) : (
        <div className="docs-grid">
          {documents.map((doc) => {
            const status = STATUS_MAP[doc.status] || STATUS_MAP.processing;
            const StatusIcon = status.icon;
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
                    <StatusIcon size={10} />
                    {status.label}
                  </span>
                  <span className="text-xs text-muted">
                    {doc.chunk_count} chunks
                  </span>
                </div>

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
                      onChange={(e) => handleVisibility(doc, e.target.value)}
                      disabled={user?.role !== 'admin' && user?.role !== 'manager'}
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
            
            <div className="flex gap-md mb-md">
              <select 
                className="input-field" 
                value={analysisType} 
                onChange={(e) => setAnalysisType(e.target.value)}
                disabled={analysisLoading}
              >
                <option value="summary">Résumé complet</option>
                <option value="risks">Analyse des risques</option>
                <option value="deadlines">Échéances & Délais</option>
                <option value="financials">Aspects financiers</option>
                <option value="action_items">Actions requises</option>
              </select>
              <button 
                className="btn btn-primary" 
                onClick={handleAnalyze}
                disabled={analysisLoading}
                style={{ whiteSpace: 'nowrap' }}
              >
                {analysisLoading ? <div className="spinner" style={{ width: '16px', height: '16px', borderWidth: '2px' }} /> : <Search size={16} />}
                Lancer l'analyse
              </button>
            </div>

            <div className="analysis-result-container card flex-col" style={{ flexGrow: 1, overflowY: 'auto', background: 'var(--bg-input)' }}>
              {analysisLoading ? (
                <div className="flex-col items-center justify-center gap-md py-lg" style={{ color: 'var(--text-muted)' }}>
                  <div className="spinner spinner-lg" />
                  <p>Analyse de l'intégralité du document en cours...</p>
                  <p className="text-xs">Cela peut prendre jusqu'à 30 secondes pour les documents volumineux.</p>
                </div>
              ) : analysisResult ? (
                <div className="markdown-body" style={{ whiteSpace: 'pre-wrap', lineHeight: '1.6' }}>
                  {analysisResult}
                </div>
              ) : (
                <div className="flex items-center justify-center" style={{ height: '100%', color: 'var(--text-muted)' }}>
                  Sélectionnez un type d'analyse et lancez l'extraction.
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
