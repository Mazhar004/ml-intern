/**
 * Lightweight localStorage persistence for UIMessage arrays,
 * keyed by session ID.
 *
 * Uses the same storage namespace (`hf-agent-messages`) that the
 * old Zustand-based store used, so existing data is compatible.
 */
import type { UIMessage } from 'ai';
import {
  pruneSessionCache,
  UI_MESSAGES_CACHE_KEY,
  writeSessionCache,
} from '@/lib/safeStorage';

const STORAGE_KEY = UI_MESSAGES_CACHE_KEY;

type MessagesMap = Record<string, UIMessage[]>;

function readAll(): MessagesMap {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    // Legacy format was { messagesBySession: {...} }
    if (parsed.messagesBySession) return parsed.messagesBySession;
    // New flat format
    if (typeof parsed === 'object' && !Array.isArray(parsed)) return parsed;
    return {};
  } catch {
    return {};
  }
}

// Best-effort: bounded to recent sessions, and never throws on a full quota.
function writeAll(map: MessagesMap, sessionId?: string): void {
  writeSessionCache(STORAGE_KEY, map, sessionId);
}

export function loadMessages(sessionId: string): UIMessage[] {
  const map = readAll();
  const messages = map[sessionId] ?? [];
  return messages;
}

/** Ids of the sessions with at least one cached message, in one read. */
export function cachedMessageSessionIds(): Set<string> {
  const map = readAll();
  return new Set(
    Object.keys(map).filter((id) => Array.isArray(map[id]) && map[id].length > 0),
  );
}

export function saveMessages(sessionId: string, messages: UIMessage[]): void {
  const map = readAll();
  // Re-insert so key order stays least recently written first.
  delete map[sessionId];
  map[sessionId] = messages;
  writeAll(map, sessionId);
}

export function deleteMessages(sessionId: string): void {
  const map = readAll();
  delete map[sessionId];
  writeAll(map);
}

export function moveMessages(fromId: string, toId: string): void {
  const map = readAll();
  if (!map[fromId]) return;
  map[toId] = map[fromId];
  delete map[fromId];
  writeAll(map, toId);
}

// Shrink a cache written before the size limits existed, and flatten the
// legacy format (once per page load).
pruneSessionCache(STORAGE_KEY, readAll());
