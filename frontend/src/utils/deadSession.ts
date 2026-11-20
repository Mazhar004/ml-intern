/**
 * What to do with a chat whose backend session is gone: GET /api/session/<id>
 * answered 404, e.g. after a backend restart, or for a saved chat that only
 * exists on disk (saved chats are never live).
 *
 * Pure (no store or storage access) so the rule can be unit-tested.
 */
import type { SessionMeta } from '@/types/agent';

/** The row fields the decision reads. */
export type DeadSessionRow = Pick<
  SessionMeta,
  'id' | 'expired' | 'sourceArchiveId' | 'archivePreview' | 'messageCount'
>;

export interface DeadSessionContext {
  /** The browser still caches messages for this chat. */
  hasClientCache: boolean;
  /** Ids from the last successful GET /api/sessions/archived (chats saved on
   *  disk), or null while that list has not loaded: then any chat may be on disk. */
  archivedSessionIds: ReadonlySet<string> | null;
}

/**
 * 'keep': keep the row and mark it expired, so its banner offers Catch me up.
 * 'drop': delete the row. 'ignore': the row is not in the list.
 */
export type DeadSessionAction = 'keep' | 'drop' | 'ignore';

/** True when the chat is, or may be, saved on disk. */
function mayBeOnDisk(row: DeadSessionRow, archivedIds: ReadonlySet<string> | null): boolean {
  // Already expired: a saved chat, or one already offered Catch me up.
  if (row.expired) return true;
  // Hydrated from disk, or restored from it by Catch me up. A row with a
  // sourceArchiveId is kept whether or not that id is in the latest list.
  if (row.sourceArchiveId || row.archivePreview !== undefined) return true;
  if ((row.messageCount ?? 0) > 0) return true;
  return archivedIds === null || archivedIds.has(row.id);
}

/**
 * Keep every chat that has, or may have, something to restore: cached
 * messages, or a copy on disk. Drop only a fresh live chat with neither
 * (typically one created just before the backend restarted). Dropping more
 * made opening saved chats delete them one after another: each deletion
 * opened the next saved chat, which 404'd in turn.
 */
export function deadSessionAction(
  row: DeadSessionRow | undefined,
  context: DeadSessionContext,
): DeadSessionAction {
  if (!row) return 'ignore';
  return context.hasClientCache || mayBeOnDisk(row, context.archivedSessionIds) ? 'keep' : 'drop';
}
