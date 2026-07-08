import { apiClient } from '../../shared/api/apiClient';

const API_BASE = (import.meta.env.VITE_API_URL || '/api').replace(/\/$/, '');

export const queryApi = {
  ask: (question: string, k: number = 3, source_filter?: string[], universe_id?: string) =>
    apiClient.post('/query', { question, k, source_filter, universe_id }),
  stream: (question: string, k: number = 3, source_filter?: string[], universe_id?: string) =>
    fetch(`${API_BASE}/query/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${localStorage.getItem('access_token') || ''}`,
      },
      body: JSON.stringify({ question, k, source_filter, universe_id }),
    }),
};
