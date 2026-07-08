import { apiClient } from '../../shared/api/apiClient';
import type { AxiosProgressEvent } from 'axios';

export const documentsApi = {
  list: () => apiClient.get('/documents'),
  get: (id: string) => apiClient.get(`/documents/${id}`),
  upload: (
    file: File,
    universe_id?: string,
    visibility: 'private' | 'department' = 'department',
    onUploadProgress?: (event: AxiosProgressEvent) => void,
  ) => {
    const form = new FormData();
    form.append('file', file);
    if (universe_id) form.append('universe_id', universe_id);
    form.append('visibility', visibility);
    return apiClient.post('/documents/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress,
    });
  },
  updateVisibility: (id: string, visibility: 'private' | 'department') =>
    apiClient.patch(`/documents/${id}/visibility`, { visibility }),
  delete: (id: string) => apiClient.delete(`/documents/${id}`),
  analyze: (id: string, analysis_type: string, prompt?: string) =>
    apiClient.post(`/documents/${id}/analyze`, { analysis_type, prompt }),
};
