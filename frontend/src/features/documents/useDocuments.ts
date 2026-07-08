import { useState, useEffect, useCallback, useRef } from 'react';
import { documentsApi } from './documentsApi';
import type { Document } from '../../types';

type UploadMsg = { type: 'success' | 'error'; text: string } | null;
type UploadVisibility = 'private' | 'department';
export type UploadProgress = {
  filename: string;
  phase: 'uploading' | 'queued' | 'processing' | 'complete' | 'failed';
  percent: number;
  documentId?: string;
} | null;

function progressFromStatus(doc: Document) {
  if (doc.status === 'indexed') return { phase: 'complete' as const, percent: 100 };
  if (doc.status === 'failed') return { phase: 'failed' as const, percent: 100 };
  if (doc.status === 'processing') {
    return { phase: 'processing' as const, percent: doc.chunk_count > 0 ? 88 : 70 };
  }
  return { phase: 'queued' as const, percent: 45 };
}

export function useDocuments() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState<UploadMsg>(null);
  const [uploadProgress, setUploadProgress] = useState<UploadProgress>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const pollTimerRef = useRef<number | null>(null);

  const fetchDocs = useCallback(async (options?: { silent?: boolean }) => {
    if (!options?.silent) setLoading(true);
    try {
      const res = await documentsApi.list();
      const nextDocs = res.data as Document[];
      setDocuments(nextDocs);
      return nextDocs;
    } catch {
      setDocuments([]);
      return [];
    } finally {
      if (!options?.silent) setLoading(false);
    }
  }, []);

  useEffect(() => { void fetchDocs(); }, [fetchDocs]);

  const upload = useCallback(async (file: File, visibility: UploadVisibility = 'department', universe_id?: string) => {
    setUploading(true);
    setUploadMsg(null);
    setUploadProgress({
      filename: file.name,
      phase: 'uploading',
      percent: 0,
    });

    try {
      const res = await documentsApi.upload(file, universe_id, visibility, (event) => {
        if (!event.total) return;
        const percent = Math.min(99, Math.round((event.loaded * 100) / event.total));
        setUploadProgress({
          filename: file.name,
          phase: 'uploading',
          percent,
        });
      });

      const uploadedDoc = res.data as Document;
      const progress = progressFromStatus(uploadedDoc);
      setUploadProgress({
        filename: uploadedDoc.filename || file.name,
        phase: progress.phase,
        percent: progress.percent,
        documentId: uploadedDoc.id,
      });
      setUploadMsg({ type: 'success', text: `${file.name} importe - indexation en cours` });
      void fetchDocs({ silent: true });
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      const status = err.response?.status;
      setUploadProgress((current) => current ? { ...current, phase: 'failed', percent: 100 } : null);
      setUploadMsg({
        type: 'error',
        text:
          detail ||
          (status === 413
            ? "Fichier trop volumineux pour la passerelle web. Reduisez sa taille ou augmentez la limite d'upload."
            : `Erreur lors de l'import${status ? ` (HTTP ${status})` : ''}`),
      });
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = '';
    }
  }, [fetchDocs]);

  useEffect(() => {
    const hasActiveDocuments = documents.some((doc) => doc.status === 'queued' || doc.status === 'processing');
    if (!hasActiveDocuments && !uploadProgress?.documentId) {
      if (pollTimerRef.current) {
        window.clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
      return;
    }

    if (pollTimerRef.current) return;

    pollTimerRef.current = window.setInterval(async () => {
      const latestDocs = await fetchDocs({ silent: true });
      if (!uploadProgress?.documentId) return;

      const currentDoc = latestDocs.find((doc) => doc.id === uploadProgress.documentId);
      if (!currentDoc) return;

      const progress = progressFromStatus(currentDoc);
      setUploadProgress({
        filename: currentDoc.filename,
        phase: progress.phase,
        percent: progress.percent,
        documentId: currentDoc.id,
      });

      if (currentDoc.status === 'indexed') {
        window.setTimeout(() => setUploadProgress(null), 2500);
      }
    }, 2500);

    return () => {
      if (pollTimerRef.current) {
        window.clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
  }, [documents, fetchDocs, uploadProgress?.documentId]);

  const remove = useCallback(async (doc: Document) => {
    try {
      await documentsApi.delete(doc.id);
      void fetchDocs({ silent: true });
      return true;
    } catch {
      return false;
    }
  }, [fetchDocs]);

  const analyze = useCallback(async (docId: string, analysisType: string) => {
    const res = await documentsApi.analyze(docId, analysisType);
    return res.data.answer as string;
  }, []);

  const updateVisibility = useCallback(async (doc: Document, visibility: UploadVisibility) => {
    await documentsApi.updateVisibility(doc.id, visibility);
    void fetchDocs({ silent: true });
  }, [fetchDocs]);

  return {
    documents,
    loading,
    uploading,
    uploadMsg,
    uploadProgress,
    fileRef,
    fetchDocs,
    upload,
    remove,
    analyze,
    updateVisibility,
  };
}
