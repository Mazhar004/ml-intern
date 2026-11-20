/**
 * localStorage cache of raw backend (litellm Message) dicts keyed by
 * session ID. Used to restore a session into a fresh backend after the
 * Space restarts — the browser-side UIMessages are what the user sees,
 * but the LLM needs the backend format to continue the conversation.
 */
import {
  BACKEND_MESSAGES_CACHE_KEY,
  pruneSessionCache,
  writeSessionCache,
} from '@/lib/safeStorage';

const STORAGE_KEY = BACKEND_MESSAGES_CACHE_KEY;

type MessagesMap = Record<string, unknown[]>;

function readAll(): MessagesMap {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
      return parsed as MessagesMap;
    }
    return {};
  } catch {
    return {};
  }
}

// Best-effort: bounded to recent sessions, and never throws on a full quota.
function writeAll(map: MessagesMap, sessionId?: string): void {
  writeSessionCache(STORAGE_KEY, map, sessionId);
}

export function loadBackendMessages(sessionId: string): unknown[] {
  const map = readAll();
  return map[sessionId] ?? [];
}

/** Ids of the sessions with at least one cached message, in one read. */
export function cachedBackendSessionIds(): Set<string> {
  const map = readAll();
  return new Set(
    Object.keys(map).filter((id) => Array.isArray(map[id]) && map[id].length > 0),
  );
}

export function saveBackendMessages(sessionId: string, messages: unknown[]): void {
  const map = readAll();
  // Re-insert so key order stays least recently written first.
  delete map[sessionId];
  map[sessionId] = messages;
  writeAll(map, sessionId);
}

export function moveBackendMessages(fromId: string, toId: string): void {
  const map = readAll();
  if (!map[fromId]) return;
  map[toId] = map[fromId];
  delete map[fromId];
  writeAll(map, toId);
}

export function deleteBackendMessages(sessionId: string): void {
  const map = readAll();
  delete map[sessionId];
  writeAll(map);
}

// Shrink a cache written before the size limits existed (once per page load).
pruneSessionCache(STORAGE_KEY, readAll());
