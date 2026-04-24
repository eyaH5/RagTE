import { apiClient } from '../../shared/api/apiClient';

export const authApi = {
  login: (email: string, password: string) =>
    apiClient.post('/auth/login', { email, password }),
  refresh: (refresh_token: string) =>
    apiClient.post('/auth/refresh', { refresh_token }),
  logout: (refresh_token: string) =>
    apiClient.post('/auth/logout', { refresh_token }),
  me: () => apiClient.get('/auth/me'),
};
