import { createContext, useContext, useState, useEffect, type ReactNode } from 'react';
import { authApi } from './features/auth';
import type { User } from './types';

interface AuthState {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthState | null>(null);

function isValidUser(value: unknown): value is User {
  if (!value || typeof value !== 'object') return false;

  const candidate = value as Partial<User>;
  return (
    typeof candidate.id === 'string' &&
    typeof candidate.email === 'string' &&
    typeof candidate.name === 'string' &&
    typeof candidate.department_id === 'string' &&
    typeof candidate.role === 'string' &&
    typeof candidate.is_active === 'boolean'
  );
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Check for existing session on mount
  useEffect(() => {
    const token = localStorage.getItem('access_token');
    if (token) {
      authApi.me()
        .then((res) => setUser(res.data))
        .catch(() => {
          localStorage.clear();
          setUser(null);
        })
        .finally(() => setIsLoading(false));
    } else {
      setIsLoading(false);
    }
  }, []);

  const login = async (email: string, password: string) => {
    const res = await authApi.login(email, password);

    const data = res.data as {
      access_token?: unknown;
      refresh_token?: unknown;
      user?: unknown;
    };

    if (
      typeof data.access_token !== 'string' ||
      typeof data.refresh_token !== 'string' ||
      !isValidUser(data.user)
    ) {
      throw new Error('Reponse de connexion invalide. Verifiez le proxy /api et la configuration du frontend.');
    }

    const { access_token, refresh_token } = data;
    const userData = data.user;
    localStorage.setItem('access_token', access_token);
    localStorage.setItem('refresh_token', refresh_token);
    setUser(userData);
  };

  const logout = async () => {
    const refreshToken = localStorage.getItem('refresh_token');
    if (refreshToken) {
      try {
        await authApi.logout(refreshToken);
      } catch (err) {
        console.error("Failed to revoke token on backend", err);
      }
    }
    localStorage.clear();
    setUser(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated: !!user,
        login,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
