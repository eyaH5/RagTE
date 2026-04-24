import { useState, useEffect, useCallback } from 'react';
import { adminApi } from './adminApi';
import type { User, Department } from '../../types';

type AdminTab = 'users' | 'departments' | 'audit';

export function useAdmin() {
  const [activeTab, setActiveTab] = useState<AdminTab>('users');
  const [users, setUsers] = useState<User[]>([]);
  const [departments, setDepartments] = useState<Department[]>([]);
  const [auditLogs, setAuditLogs] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      if (activeTab === 'users') {
        const res = await adminApi.users();
        setUsers(res.data);
      } else if (activeTab === 'departments') {
        const res = await adminApi.departments();
        setDepartments(res.data);
      } else if (activeTab === 'audit') {
        const res = await adminApi.audit();
        setAuditLogs(res.data);
      }
    } catch (err) {
      console.error('Failed to fetch admin data', err);
    } finally {
      setLoading(false);
    }
  }, [activeTab]);

  useEffect(() => { fetchData(); }, [fetchData]);

  const createUser = useCallback(async (userData: {
    name: string;
    email: string;
    password: string;
    department_id: string;
    role: string;
  }) => {
    await adminApi.createUser(userData);
    fetchData();
  }, [fetchData]);

  return {
    activeTab,
    setActiveTab,
    users,
    departments,
    auditLogs,
    loading,
    fetchData,
    createUser,
  };
}
