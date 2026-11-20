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
  /** True once the user renamed this chat. Auto-titling never overwrites a
   *  custom title, and the "Chat N" → "Session NN" display fallback is
   *  skipped for it. Mirrors the backend's ``title_is_custom``. */
  titleIsCustom?: boolean;
  /** First-user-message preview for rows hydrated from
   *  /api/sessions/archived. Used as the de-dup key instead of the
   *  (renamable) title so renaming never splits or merges archive groups. */
  archivePreview?: string;
  createdAt: string;
  usageWindowStartedAt?: string | null;
  isActive: boolean;
  needsAttention: boolean;
  model?: string | null;
  /** True when the backend reports this session is mid-turn (from
   *  GET /sessions). Sidebar status only: background sessions — processing
   *  or not — are NOT hydrated or streamed (only the active one is), which
   *  keeps them from refilling the active-session pool and from holding the
   *  browser's per-origin connections.
   *  Transient — never persisted to localStorage (always re-derived from the
   *  server list). */
  isProcessing?: boolean;
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
   *  not the full path). Set by the New Task project picker, from the
   *  server's session list (``project``), and from archived sessions on
   *  disk when the field was saved. The sidebar groups chats by it. */
  project?: string;
  /** ISO datetime when the disk snapshot was last written (only for
   *  archived sessions). */
  savedAt?: string;
  /** Original archive session id this row was hydrated/imported from.
   *  Set when ``mergeArchivedSessions`` adds a row from /api/sessions/archived,
   *  preserved across ``renameSession`` (used by Catch-me-up) so that
   *  Delete can also remove the original disk file under the old id. */
  sourceArchiveId?: string;
  autoApprovalEnabled?: boolean;
  autoApprovalCostCapUsd?: number | null;
  autoApprovalEstimatedSpendUsd?: number;
  autoApprovalRemainingUsd?: number | null;
}

export interface User {
  authenticated: boolean;
  username?: string;
  name?: string;
  picture?: string;
  plan?: 'free' | 'pro';
}
