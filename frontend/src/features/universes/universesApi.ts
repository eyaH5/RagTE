import { apiClient } from '../../shared/api/apiClient';

export const universesApi = {
  list: (department_id?: string, page: number = 1, limit: number = 20) =>
    apiClient.get('/universes', { params: { department_id, page, limit } }),
  get: (id: string) => apiClient.get(`/universes/${id}`),
  create: (data: { name: string; description: string; department_id: string }) =>
    apiClient.post('/universes', data),
  update: (id: string, data: { name?: string; description?: string; department_id?: string }) =>
    apiClient.put(`/universes/${id}`, data),
  delete: (id: string) => apiClient.delete(`/universes/${id}`),
  documents: (id: string) => apiClient.get(`/universes/${id}/documents`),
};
