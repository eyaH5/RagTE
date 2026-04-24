import { useState, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../AuthContext';
import { Activity, Mail, Lock, AlertCircle } from 'lucide-react';
import './Login.css';

export default function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await login(email, password);
      navigate('/chat');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Erreur de connexion');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-page">
      <div className="login-bg-grid" />
      <div className="login-card animate-fade-in">
        <div className="login-header">
          <div className="login-logo">
            <Activity size={36} />
          </div>
          <h1>Tunisie Electronique</h1>
          <p className="text-secondary">Plateforme RAG d'analyse de documents</p>
        </div>

        <form onSubmit={handleSubmit} className="login-form">
          {error && (
            <div className="login-error animate-fade-in">
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}

          <div className="login-field">
            <label className="input-label" htmlFor="login-email">Email</label>
            <div className="login-input-wrapper">
              <Mail size={18} className="login-input-icon" />
              <input
                id="login-email"
                type="email"
                className="input-field login-input"
                placeholder="votre.email@tunisie-electronique.tn"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                autoFocus
              />
            </div>
          </div>

          <div className="login-field">
            <label className="input-label" htmlFor="login-password">Mot de passe</label>
            <div className="login-input-wrapper">
              <Lock size={18} className="login-input-icon" />
              <input
                id="login-password"
                type="password"
                className="input-field login-input"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
          </div>

          <button
            type="submit"
            className="btn btn-primary btn-lg w-full"
            disabled={loading}
          >
            {loading ? <div className="spinner" /> : 'Se connecter'}
          </button>
        </form>

        <div className="login-footer">
          <span className="text-xs text-muted">
            v0.2.0 · Enterprise RAG Platform
          </span>
        </div>
      </div>
    </div>
  );
}
