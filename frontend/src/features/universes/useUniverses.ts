import { useState, useEffect, useCallback } from 'react';
import { universesApi } from './universesApi';
import { adminApi } from '../admin';
import type { Universe, Department } from '../../types';

function normalizeDepartmentLabel(value: string) {
  return value
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

function uniqueDepartments(items: Department[]) {
  const seen = new Set<string>();
  return items.filter((dept) => {
    const key = normalizeDepartmentLabel(dept.name || dept.id);
    if (!key || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export function useUniverses(user: { role: string; department_id: string } | null) {
  const [universes, setUniverses] = useState<Universe[]>([]);
  const [departments, setDepartments] = useState<Department[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeDept, setActiveDept] = useState<string | undefined>(undefined);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const uniRes = await universesApi.list(activeDept);
      setUniverses(uniRes.data.universes);
    } catch {
      setUniverses([]);
    }

    try {
      if (user?.role === 'admin') {
        const deptRes = await adminApi.departments();
        setDepartments(uniqueDepartments(deptRes.data));
      } else if (user) {
        setDepartments(uniqueDepartments([
          {
            id: user.department_id,
            name: user.department_id,
            description: '',
            color: '',
          },
        ]));
      } else {
        setDepartments([]);
      }
    } catch {
      setDepartments(user ? uniqueDepartments([{
        id: user.department_id,
        name: user.department_id,
        description: '',
        color: '',
      }]) : []);
    } finally {
      setLoading(false);
    }
  }, [activeDept, user?.role, user?.department_id]);

  useEffect(() => { fetchData(); }, [fetchData]);

  const create = useCallback(async (data: {
    name: string;
    description: string;
    department_id: string;
  }) => {
    const payload = user && user.role !== 'admin'
      ? { ...data, department_id: user.department_id }
      : data;
    await universesApi.create(payload);
    fetchData();
  }, [fetchData, user?.role, user?.department_id]);

  return {
    universes,
    departments,
    loading,
    activeDept,
    setActiveDept,
    fetchData,
    create,
  };
}
