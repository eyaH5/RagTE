import { apiClient } from '../../shared/api/apiClient';

export const queryApi = {
  ask: (question: string, k: number = 6, source_filter?: string[], universe_id?: string) =>
    apiClient.post('/query', { question, k, source_filter, universe_id }),
};
