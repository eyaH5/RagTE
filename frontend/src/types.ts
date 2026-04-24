// ── User & Auth ───────────────────────────────────────────────
export type User = {
  id: string;
  email: string;
  name: string;
  department_id: string;
  role: 'admin' | 'manager' | 'analyst' | 'viewer';
  is_active: boolean;
  last_login?: string;
}

export type TokenResponse = {
  access_token: string;
  refresh_token: string;
  user: User;
}

// ── Query ─────────────────────────────────────────────────────
export type SourceCitation = {
  source: string;
  page: string;
  score: number;
}

export type QueryResponse = {
  answer: string;
  sources: SourceCitation[];
  query_time_ms: number;
}

export type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceCitation[];
  query_time_ms?: number;
  timestamp: Date;
}

// ── Documents ─────────────────────────────────────────────────
export type Document = {
  id: string;
  filename: string;
  department_id: string;
  uploaded_by: string;
  visibility: 'private' | 'department' | 'shared' | 'restricted';
  doc_type: string;
  chunk_count: number;
  status: 'processing' | 'indexed' | 'failed';
  created_at: string;
}

// ── Department ────────────────────────────────────────────────
export type Department = {
  id: string;
  name: string;
  description: string;
  color: string;
}

// ── Universe ──────────────────────────────────────────────────
export type Universe = {
  id: string;
  name: string;
  description: string;
  department_id: string;
  created_by: string;
  status: 'active' | 'deleting';
  created_at: string;
  updated_at?: string;
  document_count: number;
  conversation_count: number;
}

export type UniverseListResponse = {
  universes: Universe[];
  total: number;
}
