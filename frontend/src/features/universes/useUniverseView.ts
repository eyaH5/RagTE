import { useState, useEffect, useCallback, useRef } from 'react';
import { universesApi } from './universesApi';
import { documentsApi } from '../documents';
import type { Universe, Document } from '../../types';

type UploadVisibility = 'private' | 'department';

export function useUniverseView(id: string | undefined) {
  const [universe, setUniverse] = useState<Universe | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const fetchUniverse = useCallback(async () => {
    if (!id) return;
    setLoading(true);
    try {
      const [uniRes, docsRes] = await Promise.all([
        universesApi.get(id),
        universesApi.documents(id),
      ]);
      setUniverse(uniRes.data);
      setDocuments(docsRes.data);
    } catch {
      setUniverse(null);
      setDocuments([]);
    } finally {
      setLoading(false);
    }
  }, [id]);

  useEffect(() => { fetchUniverse(); }, [fetchUniverse]);

  const upload = useCallback(async (file: File, visibility: UploadVisibility = 'department') => {
    if (!id) return;
    setUploading(true);
    try {
      await documentsApi.upload(file, id, visibility);
      fetchUniverse();
    } catch (err: any) {
      throw err;
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = '';
    }
  }, [id, fetchUniverse]);

  const deleteUniverse = useCallback(async () => {
    if (!id) return;
    await universesApi.delete(id);
  }, [id]);

  return {
    universe,
    documents,
    loading,
    uploading,
    fileRef,
    fetchUniverse,
    upload,
    deleteUniverse,
  };
}
