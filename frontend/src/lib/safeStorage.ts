/**
 * localStorage writes that can never break the app.
 *
 * Browsers cap localStorage at about 5M chars per origin and throw a
 * QuotaExceededError once it is full. Unguarded, that error escapes from
 * zustand's persist middleware inside set(), so an action like createSession
 * throws after its state has already changed.
 *
 * The per-session message caches are by far the largest tenants and are only
 * copies (the backend is the source of truth for messages). So a write that
 * hits the quota evicts cached sessions other than the active one, least
 * recently written first, and retries once. If the retry fails too, it logs
 * a warning and the app carries on without that write.
 */
import type { StateStorage } from 'zustand/middleware';
import { logger } from '@/utils/logger';

/** The per-session message caches. Each is a JSON object mapping session id
 *  to that session's messages, least recently written session first. */
export const UI_MESSAGES_CACHE_KEY = 'hf-agent-messages';
export const BACKEND_MESSAGES_CACHE_KEY = 'hf-agent-backend-messages';
const MESSAGE_CACHE_KEYS = [UI_MESSAGES_CACHE_KEY, BACKEND_MESSAGES_CACHE_KEY] as const;
export type MessageCacheKey = (typeof MESSAGE_CACHE_KEYS)[number];

/** Sessions one message cache keeps (the most recently written ones). */
const MAX_CACHED_SESSIONS = 8;
/** A session whose serialized messages are longer than this is not cached. */
const MAX_CACHED_SESSION_CHARS = 1_500_000;
/** Size one message cache may reach; older sessions are dropped first. */
const MAX_CACHE_CHARS = 2_000_000;
/** Space freed beyond what a failed write needs, so the retry has room. */
const RECLAIM_HEADROOM_CHARS = 64 * 1024;

const QUOTA_ERROR_NAMES = new Set(['QuotaExceededError', 'NS_ERROR_DOM_QUOTA_REACHED']);
const QUOTA_ERROR_CODES = new Set([22, 1014]);

/** A message cache as stored: session id -> that session's messages. */
export type SessionCache = Record<string, unknown>;

/** A cached session with its messages already serialized. */
interface CachedSession {
  id: string;
  json: string;
}

type WriteResult = { ok: true } | { ok: false; error: unknown };

let getActiveSessionId: () => string | null = () => null;

/** Tell eviction which session is on screen; its cached messages are kept. */
export function setActiveSessionSource(source: () => string | null): void {
  getActiveSessionId = source;
}

export function isQuotaExceededError(error: unknown): boolean {
  if (typeof error !== 'object' || error === null) return false;
  const { name, code } = error as { name?: unknown; code?: unknown };
  return (
    (typeof name === 'string' && QUOTA_ERROR_NAMES.has(name)) ||
    (typeof code === 'number' && QUOTA_ERROR_CODES.has(code))
  );
}

function readItem(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function trySetItem(key: string, value: string): WriteResult {
  try {
    localStorage.setItem(key, value);
    return { ok: true };
  } catch (error) {
    return { ok: false, error };
  }
}

function removeItem(key: string): void {
  try {
    localStorage.removeItem(key);
  } catch (error) {
    logger.warn(`Could not remove "${key}" from localStorage:`, error);
  }
}

function readSessionCache(key: string): SessionCache {
  const raw = readItem(key);
  if (!raw) return {};
  try {
    const parsed: unknown = JSON.parse(raw);
    return typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)
      ? (parsed as SessionCache)
      : {};
  } catch {
    return {};
  }
}

function serializeEach(cache: SessionCache): CachedSession[] {
  return Object.entries(cache).flatMap(([id, messages]) => {
    const json = JSON.stringify(messages);
    return json === undefined ? [] : [{ id, json }];
  });
}

/** Same output as JSON.stringify on the cache, without re-serializing. */
function serializeSessions(sessions: CachedSession[]): string {
  return `{${sessions.map(({ id, json }) => `${JSON.stringify(id)}:${json}`).join(',')}}`;
}

/** Chars a session takes up inside its serialized cache. */
function sessionChars({ id, json }: CachedSession): number {
  return JSON.stringify(id).length + json.length + 2;
}

/** Keep the most recently written sessions that fit the cache limits. */
function applyCacheLimits(sessions: CachedSession[]): CachedSession[] {
  const kept: CachedSession[] = [];
  let total = 0;
  for (const session of [...sessions].reverse()) {
    if (kept.length >= MAX_CACHED_SESSIONS) break;
    const chars = sessionChars(session);
    if (session.json.length > MAX_CACHED_SESSION_CHARS || total + chars > MAX_CACHE_CHARS) continue;
    kept.push(session);
    total += chars;
  }
  return kept.reverse();
}

/** Session ids across the caches, least recently written first. A session
 *  counts as recent as its latest relative position in any cache. */
function sessionsOldestFirst(caches: CachedSession[][], keep: ReadonlySet<string>): string[] {
  const recency = new Map<string, number>();
  for (const sessions of caches) {
    sessions.forEach(({ id }, i) => {
      recency.set(id, Math.max(recency.get(id) ?? 0, (i + 1) / sessions.length));
    });
  }
  return [...recency.keys()]
    .filter((id) => !keep.has(id))
    .sort((a, b) => (recency.get(a) ?? 0) - (recency.get(b) ?? 0));
}

/** Store a trimmed cache. Shrinking should always fit; if not, drop it. */
function rewriteCache(key: string, sessions: CachedSession[]): void {
  if (sessions.length > 0 && trySetItem(key, serializeSessions(sessions)).ok) return;
  removeItem(key);
}

/**
 * Free at least `charsNeeded` chars by evicting sessions from every message
 * cache, least recently written first, never touching ids in `keep`.
 * `pending` is a cache the caller is about to write: it is trimmed in memory
 * and returned (the other caches are rewritten here).
 */
function evictSessions(
  charsNeeded: number,
  keep: ReadonlySet<string>,
  pending?: { key: MessageCacheKey; sessions: CachedSession[] },
): CachedSession[] {
  const caches = new Map<string, CachedSession[]>(
    MESSAGE_CACHE_KEYS.map((key) => [
      key,
      key === pending?.key ? pending.sessions : serializeEach(readSessionCache(key)),
    ]),
  );
  const changed = new Set<string>();
  let freed = 0;
  for (const id of sessionsOldestFirst([...caches.values()], keep)) {
    if (freed >= charsNeeded) break;
    for (const [key, sessions] of caches) {
      const evicted = sessions.find((s) => s.id === id);
      if (!evicted) continue;
      freed += sessionChars(evicted);
      caches.set(key, sessions.filter((s) => s !== evicted));
      changed.add(key);
    }
  }
  for (const key of changed) {
    if (key !== pending?.key) rewriteCache(key, caches.get(key) ?? []);
  }
  return pending ? caches.get(pending.key) ?? [] : [];
}

function protectedSessions(sessionId?: string): Set<string> {
  return new Set([getActiveSessionId(), sessionId].filter((id): id is string => Boolean(id)));
}

/** Chars a write adds to localStorage usage. */
function growthOf(key: string, value: string): number {
  const current = readItem(key);
  return current === null ? key.length + value.length : value.length - current.length;
}

/**
 * Write `value`. On a quota error, `reclaim` frees space (it gets the chars
 * to free) and returns the value to retry with; the retry happens once.
 * Never throws; returns whether the write landed.
 */
function writeOrReclaim(
  key: string,
  value: string,
  reclaim: (charsNeeded: number) => string,
): boolean {
  const first = trySetItem(key, value);
  if (first.ok) return true;
  if (!isQuotaExceededError(first.error)) {
    logger.warn(`Could not write "${key}" to localStorage:`, first.error);
    return false;
  }
  let retryValue: string;
  try {
    retryValue = reclaim(Math.max(growthOf(key, value), 0) + RECLAIM_HEADROOM_CHARS);
  } catch (error) {
    logger.warn(`Could not free localStorage space for "${key}":`, error);
    return false;
  }
  const retry = trySetItem(key, retryValue);
  if (retry.ok) return true;
  logger.warn(`localStorage is full; skipped writing "${key}":`, retry.error);
  return false;
}

/**
 * localStorage.setItem that never throws. On a quota error it evicts cached
 * messages of sessions other than the active one and retries once.
 * Returns whether the value was stored.
 */
export function safeSetItem(key: string, value: string): boolean {
  return writeOrReclaim(key, value, (charsNeeded) => {
    evictSessions(charsNeeded, protectedSessions());
    return value;
  });
}

/**
 * Store a message cache within the cache limits (the most recently written
 * sessions win; oversized ones are not cached). On a quota error it evicts
 * other sessions, never `sessionId` or the active one, and retries once.
 */
export function writeSessionCache(
  key: MessageCacheKey,
  cache: SessionCache,
  sessionId?: string,
): boolean {
  const sessions = applyCacheLimits(serializeEach(cache));
  return writeOrReclaim(key, serializeSessions(sessions), (charsNeeded) =>
    serializeSessions(
      evictSessions(charsNeeded, protectedSessions(sessionId), { key, sessions }),
    ),
  );
}

/** Rewrite a stored cache that breaks the limits (or uses an old format). */
export function pruneSessionCache(key: MessageCacheKey, cache: SessionCache): void {
  const raw = readItem(key);
  if (raw === null) return;
  const sessions = applyCacheLimits(serializeEach(cache));
  if (serializeSessions(sessions) !== raw) rewriteCache(key, sessions);
}

/** localStorage for zustand's createJSONStorage: writes never throw. */
export const safeLocalStorage: StateStorage = {
  getItem: readItem,
  setItem: (key, value) => {
    safeSetItem(key, value);
  },
  removeItem,
};
