import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useUniverses } from '.';
import { useAuth } from '../../AuthContext';
import {
  Globe, FileText, MessageSquare, Plus, RefreshCw, Search,
} from 'lucide-react';
import './Universes.css';

const DEPT_COLORS: Record<string, string> = {
  backoffice: '#3b82f6',
  software: '#10b981',
  commerciale: '#f59e0b',
  infrastructure: '#8b5cf6',
  admin: '#6b7280',
};

export default function UniversesPage() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const {
    universes, departments, loading, activeDept,
    setActiveDept, fetchData, create,
  } = useUniverses(user);

  const [showCreate, setShowCreate] = useState(false);
  const [search, setSearch] = useState('');

  // Create form state
  const [newName, setNewName] = useState('');
  const [newDesc, setNewDesc] = useState('');
  const [newDept, setNewDept] = useState(user?.department_id || '');
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    if (!user) return;
    if (user.role !== 'admin') {
      setNewDept(user.department_id);
      return;
    }
    if (!newDept && departments.length > 0) {
      setNewDept(departments[0].id);
    }
  }, [user, departments, newDept]);

  const handleCreate = async () => {
    if (!newName.trim() || !newDept) return;
    setCreating(true);
    try {
      await create({
        name: newName.trim(),
        description: newDesc.trim(),
        department_id: newDept,
      });
      setShowCreate(false);
      setNewName('');
      setNewDesc('');
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Erreur lors de la création');
    } finally {
      setCreating(false);
    }
  };

  const filtered = universes.filter((u) =>
    u.name.toLowerCase().includes(search.toLowerCase()) ||
    u.description.toLowerCase().includes(search.toLowerCase())
  );

  const canCreate = user && ['admin', 'manager', 'analyst'].includes(user.role);

  return (
    <div className="universes-page">
      {/* ── Header ─────────────────────────────────────────── */}
      <div className="universes-header">
        <div>
          <h1>
            <Globe size={22} />
            Universes
          </h1>
          <p className="text-secondary text-sm">
            {universes.length} workspace{universes.length !== 1 ? 's' : ''} disponible{universes.length !== 1 ? 's' : ''}
          </p>
        </div>
        <div className="flex gap-sm items-center">
          <div style={{ position: 'relative' }}>
            <Search size={16} style={{
              position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)',
              color: 'var(--text-muted)',
            }} />
            <input
              className="input-field"
              style={{ paddingLeft: 36, width: 240 }}
              placeholder="Rechercher un universe..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          <button className="btn btn-ghost" onClick={fetchData} disabled={loading}>
            <RefreshCw size={16} className={loading ? 'spin-icon' : ''} />
          </button>
          {canCreate && (
            <button className="btn btn-primary" onClick={() => setShowCreate(true)}>
              <Plus size={16} />
              Créer
            </button>
          )}
        </div>
      </div>

      {/* ── Department Tabs ────────────────────────────────── */}
      {user?.role === 'admin' && (
        <div className="dept-tabs mb-md">
          <button
            className={`dept-tab ${!activeDept ? 'dept-tab--active' : ''}`}
            onClick={() => setActiveDept(undefined)}
          >
            Tous
          </button>
          {departments.map((dept) => (
            <button
              key={dept.id}
              className={`dept-tab ${activeDept === dept.id ? 'dept-tab--active' : ''}`}
              onClick={() => setActiveDept(dept.id)}
              style={activeDept === dept.id ? { background: dept.color } : undefined}
            >
              {dept.name}
            </button>
          ))}
        </div>
      )}

      {/* ── Grid ───────────────────────────────────────────── */}
      {loading ? (
        <div className="universes-loading">
          <div className="spinner spinner-lg" />
          <span>Chargement des universes…</span>
        </div>
      ) : filtered.length === 0 && !canCreate ? (
        <div className="universes-empty animate-fade-in">
          <Globe size={48} color="var(--text-muted)" />
          <h3>Aucun universe disponible</h3>
          <p className="text-secondary">Aucun workspace n'a été créé pour le moment.</p>
        </div>
      ) : (
        <div className="universes-grid">
          {/* Create card */}
          {canCreate && (
            <div
              className="card universe-card universe-card--create animate-fade-in"
              onClick={() => setShowCreate(true)}
            >
              <div className="create-card-content">
                <Plus size={32} />
                <p>Créer un Universe</p>
              </div>
            </div>
          )}

          {/* Universe cards */}
          {filtered.map((uni, i) => {
            const color = DEPT_COLORS[uni.department_id] || 'var(--brand-primary)';
            return (
              <div
                key={uni.id}
                className="card universe-card animate-fade-in"
                style={{
                  '--card-accent': color,
                  animationDelay: `${i * 50}ms`,
                } as React.CSSProperties}
                onClick={() => navigate(`/universes/${uni.id}`)}
              >
                <div className="universe-card__header">
                  <h3 className="universe-card__title">{uni.name}</h3>
                  <span
                    className="universe-card__dept"
                    style={{
                      background: `${color}20`,
                      color: color,
                    }}
                  >
                    {uni.department_id}
                  </span>
                </div>

                <p className="universe-card__desc">
                  {uni.description || 'Aucune description'}
                </p>

                <div className="universe-card__stats">
                  <div className="universe-card__stat">
                    <FileText size={14} />
                    <strong>{uni.document_count}</strong> docs
                  </div>
                  <div className="universe-card__stat">
                    <MessageSquare size={14} />
                    <strong>{uni.conversation_count}</strong> chats
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ── Create Modal ──────────────────────────────────── */}
      {showCreate && (
        <div className="modal-overlay" onClick={() => setShowCreate(false)}>
          <div className="modal animate-fade-in" onClick={(e) => e.stopPropagation()}>
            <h2>Créer un Universe</h2>

            <div className="modal-field">
              <label className="input-label">Nom</label>
              <input
                className="input-field"
                placeholder="Ex: Appel d'offre STEG 2026"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                autoFocus
              />
            </div>

            <div className="modal-field">
              <label className="input-label">Description</label>
              <textarea
                className="input-field"
                placeholder="Décrivez le contexte de ce workspace..."
                value={newDesc}
                onChange={(e) => setNewDesc(e.target.value)}
                rows={3}
                style={{ resize: 'vertical' }}
              />
            </div>

            {user?.role === 'admin' && (
              <div className="modal-field">
                <label className="input-label">Département (persona IA)</label>
                <select
                  className="input-field"
                  value={newDept}
                  onChange={(e) => setNewDept(e.target.value)}
                >
                  {departments.map((dept) => (
                    <option key={dept.id} value={dept.id}>
                      {dept.name}
                    </option>
                  ))}
                </select>
              </div>
            )}

            <div className="modal-actions">
              <button className="btn btn-ghost" onClick={() => setShowCreate(false)}>
                Annuler
              </button>
              <button
                className="btn btn-primary"
                onClick={handleCreate}
                disabled={creating || !newName.trim()}
              >
                {creating ? <div className="spinner" /> : <Plus size={16} />}
                {creating ? 'Création...' : 'Créer'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
