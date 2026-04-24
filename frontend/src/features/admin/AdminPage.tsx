import { useState } from 'react';
import { useAdmin } from '.';
import { 
  Users, 
  Building2, 
  History, 
  UserPlus, 
  ShieldCheck, 
  CheckCircle2, 
  XCircle,
  Mail,
  Briefcase
} from 'lucide-react';
import './Admin.css';

export default function AdminPage() {
  const {
    activeTab, setActiveTab,
    users, departments, auditLogs,
    createUser,
  } = useAdmin();

  const [isModalOpen, setIsModalOpen] = useState(false);

  // New user form state
  const [newUser, setNewUser] = useState({
    name: '',
    email: '',
    password: '',
    department_id: 'backoffice',
    role: 'viewer'
  });

  const handleCreateUser = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await createUser(newUser);
      setIsModalOpen(false);
      setNewUser({ name: '', email: '', password: '', department_id: 'backoffice', role: 'viewer' });
    } catch (err: any) {
      alert(err.response?.data?.detail || "Erreur lors de la création");
    }
  };

  return (
    <div className="admin-page">
      <header className="admin-header">
        <div className="admin-header__info">
          <h1>Administration Système</h1>
          <p>Gérez les accès, les départements et surveillez l'activité.</p>
        </div>
        <div className="admin-tabs">
          <button 
            className={`admin-tab ${activeTab === 'users' ? 'admin-tab--active' : ''}`}
            onClick={() => setActiveTab('users')}
          >
            <Users size={18} />
            Utilisateurs
          </button>
          <button 
            className={`admin-tab ${activeTab === 'departments' ? 'admin-tab--active' : ''}`}
            onClick={() => setActiveTab('departments')}
          >
            <Building2 size={18} />
            Départements
          </button>
          <button 
            className={`admin-tab ${activeTab === 'audit' ? 'admin-tab--active' : ''}`}
            onClick={() => setActiveTab('audit')}
          >
            <History size={18} />
            Audit Log
          </button>
        </div>
      </header>

      <main className="admin-content">
        {activeTab === 'users' && (
          <section className="admin-section">
            <div className="section-header">
              <h2>Gestion des Utilisateurs</h2>
              <button className="btn btn-primary" onClick={() => setIsModalOpen(true)}>
                <UserPlus size={18} />
                Nouvel Utilisateur
              </button>
            </div>

            <div className="table-container">
              <table className="admin-table">
                <thead>
                  <tr>
                    <th>Utilisateur</th>
                    <th>Email</th>
                    <th>Département</th>
                    <th>Rôle</th>
                    <th>Statut</th>
                    <th>Dernière Connexion</th>
                  </tr>
                </thead>
                <tbody>
                  {users.map(user => (
                    <tr key={user.id}>
                      <td>
                        <div className="user-cell">
                          <div className="user-avatar" style={{ background: 'var(--brand-primary)' }}>
                            {user.name.charAt(0)}
                          </div>
                          <span>{user.name}</span>
                        </div>
                      </td>
                      <td>{user.email}</td>
                      <td>
                        <span className="badge badge-outline">{user.department_id}</span>
                      </td>
                      <td>
                        <span className={`role-tag role-${user.role}`}>
                          <ShieldCheck size={14} />
                          {user.role}
                        </span>
                      </td>
                      <td>
                        {user.is_active ? (
                          <span className="status-indicator status-active">
                            <CheckCircle2 size={14} /> Actif
                          </span>
                        ) : (
                          <span className="status-indicator status-inactive">
                            <XCircle size={14} /> Inactif
                          </span>
                        )}
                      </td>
                      <td className="text-muted">
                        {user.last_login ? new Date(user.last_login).toLocaleDateString() : 'Jamais'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {activeTab === 'departments' && (
          <section className="admin-section">
            <div className="section-header">
              <h2>Départements</h2>
            </div>
            <div className="dept-grid">
              {departments.map(dept => (
                <div key={dept.id} className="dept-card" style={{ borderTop: `4px solid ${dept.color}` }}>
                  <h3>{dept.name}</h3>
                  <p>{dept.description}</p>
                  <span className="dept-id">ID: {dept.id}</span>
                </div>
              ))}
            </div>
          </section>
        )}

        {activeTab === 'audit' && (
          <section className="admin-section">
            <div className="section-header">
              <h2>Journal d'Audit</h2>
            </div>
            <div className="audit-list">
              {auditLogs.map(log => (
                <div key={log.id} className="audit-item">
                  <div className="audit-time">{new Date(log.created_at).toLocaleString()}</div>
                  <div className="audit-action">
                    <span className="badge">{log.action}</span>
                  </div>
                  <div className="audit-details">
                    User: <strong>{log.user_id}</strong> | Resource: <code>{log.resource}</code>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}
      </main>

      {/* Create User Modal */}
      {isModalOpen && (
        <div className="modal-overlay">
          <div className="modal-card">
            <h2>Créer un utilisateur</h2>
            <form onSubmit={handleCreateUser}>
              <div className="form-group">
                <label><Users size={16} /> Nom Complet</label>
                <input 
                  type="text" 
                  value={newUser.name} 
                  onChange={e => setNewUser({...newUser, name: e.target.value})}
                  required
                />
              </div>
              <div className="form-group">
                <label><Mail size={16} /> Email Professional</label>
                <input 
                  type="email" 
                  value={newUser.email} 
                  onChange={e => setNewUser({...newUser, email: e.target.value})}
                  required
                />
              </div>
              <div className="form-group">
                <label>Mot de passe (min 8 chars)</label>
                <input 
                  type="password" 
                  value={newUser.password} 
                  onChange={e => setNewUser({...newUser, password: e.target.value})}
                  required
                />
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label><Briefcase size={16} /> Département</label>
                  <select 
                    value={newUser.department_id} 
                    onChange={e => setNewUser({...newUser, department_id: e.target.value})}
                  >
                    <option value="backoffice">Back Office</option>
                    <option value="software">Développement Logiciel</option>
                    <option value="commerciale">Commerciale</option>
                    <option value="infrastructure">Infrastructure</option>
                    <option value="admin">Admin</option>
                  </select>
                </div>
                <div className="form-group">
                  <label><ShieldCheck size={16} /> Rôle</label>
                  <select 
                    value={newUser.role} 
                    onChange={e => setNewUser({...newUser, role: e.target.value})}
                  >
                    <option value="viewer">Viewer</option>
                    <option value="analyst">Analyst</option>
                    <option value="manager">Manager</option>
                    <option value="admin">Admin</option>
                  </select>
                </div>
              </div>
              <div className="modal-actions">
                <button type="button" className="btn btn-ghost" onClick={() => setIsModalOpen(false)}>Annuler</button>
                <button type="submit" className="btn btn-primary">Créer le compte</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
