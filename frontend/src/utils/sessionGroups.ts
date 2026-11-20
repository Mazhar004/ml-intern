/**
 * Group sidebar chats by the project folder they are bound to.
 *
 * Pure helpers (no React, no storage access) so the grouping, the project
 * filter fallback, and the stored-preference parsing can be unit-tested.
 */
import type { SessionMeta } from '@/types/agent';

/** Filter value that shows every group. */
export const ALL_PROJECTS = 'all';
/** Group key (and filter value) for chats not bound to a project folder. */
export const NO_PROJECT_KEY = 'no-project';
/** Every project's key carries this prefix, so no folder name can collide
 *  with ALL_PROJECTS or NO_PROJECT_KEY. */
const PROJECT_KEY_PREFIX = 'project:';

/** The row fields grouping reads. */
export type GroupableSession = Pick<SessionMeta, 'project' | 'savedAt' | 'createdAt'>;

export interface ProjectGroup<T extends GroupableSession = SessionMeta> {
  /** Stable id: `project:<folder name>`, or NO_PROJECT_KEY. */
  key: string;
  /** Folder name (trimmed, case preserved); null for the "No project" group. */
  project: string | null;
  /** The group's rows, in the order they were passed in. */
  sessions: readonly T[];
  count: number;
  /** Epoch ms of the group's most recent chat; -Infinity when none has a valid time. */
  latestAt: number;
}

/** The project folder name, or null when the value is not a non-blank string. */
export function normalizeProjectName(value: unknown): string | null {
  if (typeof value !== 'string') return null;
  return value.trim() || null;
}

/**
 * The folder a typed project name is filed under. Mirrors the backend
 * (backend/session_manager.py), which swaps "/" and ".." for "-" and then
 * lists the session under that folder's name.
 */
export function projectFolderName(typed: string): string {
  return typed.trim().replace(/\//g, '-').replace(/\.\./g, '-');
}

/** Group key for a project folder name (null: the "No project" group). */
export function projectGroupKey(project: string | null): string {
  return project === null ? NO_PROJECT_KEY : `${PROJECT_KEY_PREFIX}${project}`;
}

/** The same time the sidebar row shows: when it was last saved, else created. */
function sessionTime(session: GroupableSession): number {
  const stamp = session.savedAt || session.createdAt;
  const ms = stamp ? Date.parse(stamp) : Number.NaN;
  return Number.isNaN(ms) ? Number.NEGATIVE_INFINITY : ms;
}

/** Newest first; ties keep their first-appearance order. */
function byLatestChat<T extends GroupableSession>(
  a: { group: ProjectGroup<T>; firstIndex: number },
  b: { group: ProjectGroup<T>; firstIndex: number },
): number {
  if (a.group.latestAt !== b.group.latestAt) {
    return a.group.latestAt > b.group.latestAt ? -1 : 1;
  }
  return a.firstIndex - b.firstIndex;
}

/**
 * Split `rows` (in sidebar order) into one group per project folder.
 *
 * Groups are ordered by their most recent chat, newest first, with the
 * "No project" group always last. Rows keep their incoming order inside a
 * group and are the same objects that were passed in. Never mutates `rows`.
 */
export function groupSessionsByProject<T extends GroupableSession>(
  rows: readonly T[],
): ProjectGroup<T>[] {
  // Buckets in order of first appearance; built locally, so pushing is safe.
  const buckets: { key: string; project: string | null; members: T[] }[] = [];
  const bucketByKey = new Map<string, (typeof buckets)[number]>();
  for (const row of rows) {
    const project = normalizeProjectName(row.project);
    const key = projectGroupKey(project);
    const bucket = bucketByKey.get(key);
    if (bucket) {
      bucket.members.push(row);
    } else {
      const created = { key, project, members: [row] };
      bucketByKey.set(key, created);
      buckets.push(created);
    }
  }

  const ranked = buckets.map(({ key, project, members }, firstIndex) => {
    const group: ProjectGroup<T> = {
      key,
      project,
      sessions: members,
      count: members.length,
      latestAt: members.reduce(
        (latest, member) => Math.max(latest, sessionTime(member)),
        Number.NEGATIVE_INFINITY,
      ),
    };
    return { group, firstIndex };
  });

  const named = ranked.filter(({ group }) => group.project !== null).sort(byLatestChat);
  const unfiled = ranked.filter(({ group }) => group.project === null);
  return [...named, ...unfiled].map(({ group }) => group);
}

/** The row fields the approval count reads. */
type ApprovalRow = Pick<SessionMeta, 'id' | 'needsAttention'>;

/**
 * How many chats in each group wait for a tool approval, by group key. The
 * open chat is left out (it shows the approval prompt itself), and so are
 * groups with none: a missing key means zero.
 */
export function countWaitingByGroup(
  groups: readonly { key: string; sessions: readonly ApprovalRow[] }[],
  activeSessionId: string | null,
): ReadonlyMap<string, number> {
  const entries = groups
    .map((group): [string, number] => [
      group.key,
      group.sessions.filter((s) => s.needsAttention && s.id !== activeSessionId).length,
    ])
    .filter(([, waiting]) => waiting > 0);
  return new Map(entries);
}

/** Spoken name of a project group or filter option, e.g. "ISO 42001, 3 chats, 1 waiting for approval". */
export function describeProjectGroup(label: string, count: number, waiting = 0): string {
  const chats = `${count} chat${count === 1 ? '' : 's'}`;
  return waiting > 0 ? `${label}, ${chats}, ${waiting} waiting for approval` : `${label}, ${chats}`;
}

/**
 * The filter to apply: the remembered choice while its group still exists,
 * otherwise ALL_PROJECTS. With no project groups there is nothing to filter.
 */
export function resolveProjectFilter(
  stored: string | null,
  groups: readonly Pick<ProjectGroup, 'key' | 'project'>[],
): string {
  if (!stored || !groups.some((group) => group.project !== null)) return ALL_PROJECTS;
  return groups.some((group) => group.key === stored) ? stored : ALL_PROJECTS;
}

/** Group keys stored as a JSON list; missing or corrupt storage reads as none. */
export function parseGroupKeys(raw: string | null): string[] {
  if (!raw) return [];
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((key): key is string => typeof key === 'string' && key !== '');
  } catch {
    return [];
  }
}
