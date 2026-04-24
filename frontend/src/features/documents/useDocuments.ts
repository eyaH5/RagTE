import { useState, useEffect, useCallback, useRef } from 'react';
import { documentsApi } from './documentsApi';
import type { Document } from '../../types';

type UploadMsg = { type: 'success' | 'error'; text: string } | null;

export function useDocuments() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState<UploadMsg>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const fetchDocs = useCallback(async () => {
    setLoading(true);
    try {
      const res = await documentsApi.list();
      setDocuments(res.data);
    } catch {
      setDocuments([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchDocs(); }, [fetchDocs]);

  const upload = useCallback(async (file: File, universe_id?: string) => {
    setUploading(true);
    setUploadMsg(null);
    try {
      await documentsApi.upload(file, universe_id);
      setUploadMsg({ type: 'success', text: `${file.name} importé — indexation en cours` });
      fetchDocs();
    } catch (err: any) {
      setUploadMsg({
        type: 'error',
        text: err.response?.data?.detail || "Erreur lors de l'import",
      });
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = '';
    }
  }, [fetchDocs]);

  const remove = useCallback(async (doc: Document) => {
    try {
      await documentsApi.delete(doc.id);
      fetchDocs();
      return true;
    } catch {
      return false;
    }
  }, [fetchDocs]);

  const analyze = useCallback(async (docId: string, analysisType: string) => {
    const res = await documentsApi.analyze(docId, analysisType);
    return res.data.answer as string;
  }, []);

  const updateVisibility = useCallback(async (doc: Document, visibility: string) => {
    await documentsApi.updateVisibility(doc.id, visibility);
    fetchDocs();
  }, [fetchDocs]);

  return {
    documents,
    loading,
    uploading,
    uploadMsg,
    fileRef,
    fetchDocs,
    upload,
    remove,
    analyze,
    updateVisibility,
  };
}
