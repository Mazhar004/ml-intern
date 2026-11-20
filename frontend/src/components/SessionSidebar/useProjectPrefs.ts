/**
 * Sidebar project preferences, remembered in localStorage: which project the
 * chat list is filtered to, and which project groups are collapsed.
 * Missing, blocked or corrupt storage falls back to the defaults.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { safeGetItem, safeSetItem } from '@/lib/safeStorage';
import {
  ALL_PROJECTS,
  parseGroupKeys,
  resolveProjectFilter,
  type ProjectGroup,
} from '@/utils/sessionGroups';

const FILTER_STORAGE_KEY = 'ml-intern.sidebar.projectFilter';
const COLLAPSED_STORAGE_KEY = 'ml-intern.sidebar.collapsedProjects';

type GroupKeys = ReadonlySet<string>;

/**
 * The project filter to apply (a group key or ALL_PROJECTS), a setter that
 * remembers the choice, and `revealGroup`, which switches an active filter
 * that would hide the given group (e.g. a just-created chat's project).
 *
 * A remembered project with no chats shows all projects. Once `groupsSettled`
 * (the server's session lists have been merged) it is also forgotten, so it
 * can't silently come back if a chat with that project name shows up later.
 * Before that, the server may still restore the group, so it is kept.
 */
export function useProjectFilter(
  groups: readonly Pick<ProjectGroup, 'key' | 'project'>[],
  groupsSettled: boolean,
): {
  projectFilter: string;
  selectProjectFilter: (filter: string) => void;
  revealGroup: (groupKey: string) => void;
} {
  const [stored, setStored] = useState(() => safeGetItem(FILTER_STORAGE_KEY) ?? ALL_PROJECTS);
  const projectFilter = resolveProjectFilter(stored, groups);

  const selectProjectFilter = useCallback((next: string) => {
    setStored(next);
    safeSetItem(FILTER_STORAGE_KEY, next);
  }, []);

  useEffect(() => {
    if (groupsSettled && projectFilter !== stored) selectProjectFilter(projectFilter);
  }, [groupsSettled, projectFilter, stored, selectProjectFilter]);

  // Read at call time: callers run after awaits, when a captured value may be stale.
  const latestFilter = useRef(projectFilter);
  useEffect(() => {
    latestFilter.current = projectFilter;
  }, [projectFilter]);

  const revealGroup = useCallback(
    (groupKey: string) => {
      const current = latestFilter.current;
      if (current !== ALL_PROJECTS && current !== groupKey) selectProjectFilter(groupKey);
    },
    [selectProjectFilter],
  );

  return { projectFilter, selectProjectFilter, revealGroup };
}

function withKey(keys: GroupKeys, key: string): GroupKeys {
  return new Set([...keys, key]);
}

function withoutKey(keys: GroupKeys, key: string): GroupKeys {
  return new Set([...keys].filter((k) => k !== key));
}

/**
 * Collapsed project groups (by group key) and a stable toggle. The group
 * holding the active chat opens whenever the active chat changes or moves to
 * another group (including on load); the user can still collapse it after.
 */
export function useCollapsedGroups(
  activeSessionId: string | null,
  activeGroupKey: string | null,
): {
  collapsed: GroupKeys;
  toggleGroup: (groupKey: string) => void;
} {
  const [collapsed, setCollapsed] = useState<GroupKeys>(
    () => new Set(parseGroupKeys(safeGetItem(COLLAPSED_STORAGE_KEY))),
  );

  // activeSessionId is a trigger only: a new chat in an already-collapsed
  // group changes the active chat but not its group key.
  useEffect(() => {
    if (activeGroupKey === null) return;
    setCollapsed((prev) => (prev.has(activeGroupKey) ? withoutKey(prev, activeGroupKey) : prev));
  }, [activeSessionId, activeGroupKey]);

  // Save changes only: the first run holds what was just read from storage.
  const lastSaved = useRef<string | null>(null);
  useEffect(() => {
    const json = JSON.stringify([...collapsed]);
    const previous = lastSaved.current;
    lastSaved.current = json;
    if (previous !== null && previous !== json) safeSetItem(COLLAPSED_STORAGE_KEY, json);
  }, [collapsed]);

  const toggleGroup = useCallback((groupKey: string) => {
    setCollapsed((prev) => (prev.has(groupKey) ? withoutKey(prev, groupKey) : withKey(prev, groupKey)));
  }, []);

  return { collapsed, toggleGroup };
}
