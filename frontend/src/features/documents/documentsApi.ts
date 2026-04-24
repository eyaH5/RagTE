import { apiClient } from '../../shared/api/apiClient';

export const documentsApi = {
  list: () => apiClient.get('/documents'),
  get: (id: string) => apiClient.get(`/documents/${id}`),
  upload: (file: File, universe_id?: string) => {
    const form = new FormData();
    form.append('file', file);
    if (universe_id) form.append('universe_id', universe_id);
    return apiClient.post('/documents/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  updateVisibility: (id: string, visibility: string) =>
    apiClient.patch(`/documents/${id}/visibility`, { visibility }),
  delete: (id: string) => apiClient.delete(`/documents/${id}`),
  analyze: (id: string, analysis_type: string, prompt?: string) =>
    apiClient.post(`/documents/${id}/analyze`, { analysis_type, prompt }),
};
