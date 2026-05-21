/**
 * Agent-related types.
 *
 * Message and tool-call types are now provided by the Vercel AI SDK
 * (UIMessage, UIMessagePart, etc.). Only non-SDK types remain here.
 */

/** Custom metadata attached to every UIMessage via the `metadata` field. */
export interface MessageMeta {
  createdAt?: string;
}

export interface SessionMeta {
  id: string;
  title: string;
  createdAt: string;
  isActive: boolean;
  needsAttention: boolean;
  /** True when the backend no longer recognizes this session id (e.g.
   *  after a backend restart). The UI shows a recovery banner and
   *  disables input until the user chooses to restore-with-summary or
   *  start fresh. */
  expired?: boolean;
  /** Number of messages in the archived session — populated when
   *  hydrated from /api/sessions/archived. Shown as a subtitle so the
   *  user can pick the right one to resume at a glance. */
  messageCount?: number;
  /** Project folder this session is bound to (just the folder name,
   *  not the full path). Populated for new sessions created via the
   *  project picker, and for archived sessions if the field was saved. */
  project?: string;
  /** ISO datetime when the disk snapshot was last written (only for
   *  archived sessions). */
  savedAt?: string;
  /** Original archive session id this row was hydrated/imported from.
   *  Set when ``mergeArchivedSessions`` adds a row from /api/sessions/archived,
   *  preserved across ``renameSession`` (used by Catch-me-up) so that
   *  Delete can also remove the original disk file under the old id. */
  sourceArchiveId?: string;
}

export interface ToolApproval {
  tool_call_id: string;
  approved: boolean;
  feedback?: string | null;
}

export interface User {
  authenticated: boolean;
  username?: string;
  name?: string;
  picture?: string;
  orgMember?: boolean;
}
