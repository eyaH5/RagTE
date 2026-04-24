import { NavLink, Outlet, useNavigate } from 'react-router-dom';
import { useAuth } from '../AuthContext';
import {
  MessageSquare, FileText, Shield, LogOut, Activity, ChevronLeft, ChevronRight,
  Globe,
} from 'lucide-react';
import { useState } from 'react';
import './Layout.css';

const NAV_ITEMS = [
  { to: '/universes', icon: Globe, label: 'Universes', roles: ['admin', 'manager', 'analyst', 'viewer'] },
  { to: '/chat', icon: MessageSquare, label: 'Chat', roles: ['admin', 'manager', 'analyst', 'viewer'] },
  { to: '/documents', icon: FileText, label: 'Documents', roles: ['admin', 'manager', 'analyst'] },
  { to: '/admin', icon: Shield, label: 'Administration', roles: ['admin'] },
];

const DEPT_COLORS: Record<string, string> = {
  backoffice: 'var(--dept-backoffice)',
  software: 'var(--dept-software)',
  commerciale: 'var(--dept-commerciale)',
  infrastructure: 'var(--dept-infrastructure)',
  admin: 'var(--dept-admin)',
};

export default function Layout() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [collapsed, setCollapsed] = useState(false);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const deptColor = DEPT_COLORS[user?.department_id || ''] || 'var(--brand-primary)';

  return (
    <div className="layout">
      {/* ── Sidebar ─────────────────────────────────────────── */}
      <aside className={`sidebar ${collapsed ? 'sidebar--collapsed' : ''}`}>
        <div className="sidebar__header">
          <div className="sidebar__logo">
            <Activity size={24} color="var(--brand-primary)" />
            {!collapsed && <span className="sidebar__title">TE RAG</span>}
          </div>
          <button
            className="btn btn-icon btn-ghost sidebar__toggle"
            onClick={() => setCollapsed(!collapsed)}
            aria-label="Toggle sidebar"
          >
            {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
          </button>
        </div>

        <nav className="sidebar__nav">
          {NAV_ITEMS.filter((item) => user && item.roles.includes(user.role)).map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `sidebar__link ${isActive ? 'sidebar__link--active' : ''}`
              }
            >
              <item.icon size={20} />
              {!collapsed && <span>{item.label}</span>}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar__footer">
          <div className="sidebar__user">
            <div
              className="sidebar__avatar"
              style={{ background: deptColor }}
            >
              {user?.name?.charAt(0) || '?'}
            </div>
            {!collapsed && (
              <div className="sidebar__user-info">
                <span className="sidebar__user-name">{user?.name}</span>
                <span className="sidebar__user-dept">
                  {user?.department_id} · {user?.role}
                </span>
              </div>
            )}
          </div>
          <button
            className="btn btn-icon btn-ghost"
            onClick={handleLogout}
            title="Déconnexion"
          >
            <LogOut size={18} />
          </button>
        </div>
      </aside>

      {/* ── Main Content ───────────────────────────────────── */}
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  );
}
