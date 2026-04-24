import { apiClient } from '../../shared/api/apiClient';

export const adminApi = {
  departments: () => apiClient.get('/admin/departments'),
  users: () => apiClient.get('/admin/users'),
  createUser: (userData: any) => apiClient.post('/admin/users', userData),
  audit: (limit: number = 50) => apiClient.get(`/admin/audit?limit=${limit}`),
};
