import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import type { SessionMeta } from '@/types/agent';
import {
  cachedMessageSessionIds,
  deleteMessages,
  moveMessages,
} from '@/lib/chat-message-store';
import {
  cachedBackendSessionIds,
  deleteBackendMessages,
  moveBackendMessages,
} from '@/lib/backend-message-store';
import { safeLocalStorage, setActiveSessionSource } from '@/lib/safeStorage';
import { deadSessionAction } from '@/utils/deadSession';
import { normalizeProjectName } from '@/utils/sessionGroups';

interface SessionStore {
  sessions: SessionMeta[];
  activeSessionId: string | null;
  /** Ids from the last successful GET /api/sessions/archived (chats saved on
   *  disk), or null until that list loads. Per page load: never persisted. */
  archivedSessionIds: ReadonlySet<string> | null;
  /** True once this page load has merged both server lists (GET /api/sessions
   *  and /api/sessions/archived). Until then the sidebar keeps a remembered
   *  project filter whose group has not shown up yet. Never persisted. */
  serverListsMerged: boolean;

  // Actions
  /** `project` is the folder picked in New Task (blank or omitted: none). */
  createSession: (id: string, model?: string | null, project?: string | null) => void;
  deleteSession: (id: string) => void;
  switchSession: (id: string) => void;
  setSessionActive: (id: string, isActive: boolean) => void;
  /** Track whether this session is mid-turn. Set false on terminal events so a
   *  finished task stops looking "processing" (which would otherwise last
   *  until the next GET /sessions merge). */
  setSessionProcessing: (id: string, isProcessing: boolean) => void;
  /** Apply an auto-generated title. No-op for chats the user renamed. */
  updateSessionTitle: (id: string, title: string) => void;
  /** Set a title and whether it is a user rename (also used to revert a
   *  failed optimistic rename). */
  setSessionTitle: (id: string, title: string, titleIsCustom: boolean) => void;
  updateSessionModel: (id: string, model: string | null) => void;
  setNeedsAttention: (id: string, needs: boolean) => void;
  /** Mark a session as expired (backend no longer has it). The UI shows a
   *  recovery banner and disables input. */
  markExpired: (id: string) => void;
  /** Clear the expired flag (used after restore-with-summary succeeds). */
  clearExpired: (id: string) => void;
  /** Merge durable server-side sessions into local sidebar metadata. */
  mergeServerSessions: (sessions: Array<{
    session_id: string;
    title?: string | null;
    title_is_custom?: boolean;
    created_at: string;
    usage_window_started_at?: string | null;
    is_active?: boolean;
    is_processing?: boolean;
    model?: string | null;
    /** Project folder name, when the session is bound to one. */
    project?: string | null;
    pending_approval?: unknown[] | null;
    auto_approval?: {
      enabled?: boolean;
      cost_cap_usd?: number | null;
      estimated_spend_usd?: number;
      remaining_usd?: number | null;
    } | null;
  }>) => void;
  updateSessionYolo: (id: string, policy: {
    enabled: boolean;
    cost_cap_usd?: number | null;
    estimated_spend_usd?: number;
    remaining_usd?: number | null;
  }) => void;
  /** Atomically swap a session's id in the list + both localStorage caches.
   *  Used when we rehydrate an expired session into a freshly-created backend
   *  session — preserves title, timestamps, and messages. */
  renameSession: (oldId: string, newId: string) => void;
  /** Merge a list of archived (server-side, on-disk) sessions into the
   *  sidebar so they're visible & clickable. Skips ids already in the store
   *  and marks newly-added ones as expired so the UI shows the recovery
   *  banner — clicking "Catch me up" then triggers a full /api/session/import
   *  rehydration. Items: {session_id, saved_at, message_count, first_user_preview,
   *  title, title_is_custom}. */
  mergeArchivedSessions: (archived: ArchivedSessionRow[]) => void;
  /** Record the ids a successful GET /api/sessions/archived returned. */
  setArchivedSessionIds: (ids: readonly string[]) => void;
  setServerListsMerged: (merged: boolean) => void;
  /** The backend has no session for this chat (it answered 404). Keeps the
   *  row, marked expired so its banner offers Catch me up, whenever it has or
   *  may have something to restore (see utils/deadSession.ts); deletes only a
   *  fresh live chat with nothing to restore. Never touches a session_logs/
   *  file. Ids not in the list are ignored. */
  resolveDeadSession: (id: string) => void;
}

interface ArchivedSessionRow {
  session_id: string;
  saved_at?: string | null;
  message_count?: number;
  first_user_preview?: string;
  project?: string | null;
  title?: string | null;
  title_is_custom?: boolean;
}

/** A user rename saved on disk wins over the local (preview/auto) title. */
function withArchivedCustomTitle(
  session: SessionMeta,
  archivedById: Map<string, ArchivedSessionRow>,
): SessionMeta {
  const row = session.expired ? archivedById.get(session.id) : undefined;
  if (!row?.title_is_custom || !row.title || row.title === session.title) return session;
  return { ...session, title: row.title, titleIsCustom: true };
}

/** Rows saved before the project was tracked locally take it from disk. */
function withArchivedProject(
  session: SessionMeta,
  archivedById: Map<string, ArchivedSessionRow>,
): SessionMeta {
  if (session.project) return session;
  const row =
    archivedById.get(session.id) ??
    (session.sourceArchiveId ? archivedById.get(session.sourceArchiveId) : undefined);
  const project = normalizeProjectName(row?.project);
  return project ? { ...session, project } : session;
}

/** Ids of the chats whose messages this browser still caches (either cache). */
function cachedMessageIds(): Set<string> {
  return new Set([...cachedBackendSessionIds(), ...cachedMessageSessionIds()]);
}

/** The browser still caches messages for this chat (in either cache). */
function hasCachedMessages(id: string): boolean {
  return cachedMessageIds().has(id);
}

/**
 * The chat to open after the open one is deleted: the last live (not
 * expired) row, which is the most recently added one because new chats are
 * appended; else the last row; else none. Store order, not createdAt: the
 * rows mix UTC ("...Z"), naive UTC and naive local times, which don't
 * compare. A saved chat is never live in the backend, so opening one would
 * only 404.
 */
function nextActiveSessionId(sessions: readonly SessionMeta[]): string | null {
  const live = sessions.filter((s) => !s.expired);
  return (live[live.length - 1] ?? sessions[sessions.length - 1])?.id ?? null;
}

function sameMembers(a: ReadonlySet<string>, b: ReadonlySet<string>): boolean {
  return a.size === b.size && [...a].every((id) => b.has(id));
}

export const useSessionStore = create<SessionStore>()(
  persist(
    (set, get) => ({
      sessions: [],
      activeSessionId: null,
      archivedSessionIds: null,
      serverListsMerged: false,

      createSession: (id: string, model?: string | null, project?: string | null) => {
        const now = new Date().toISOString();
        const newSession: SessionMeta = {
          id,
          title: `Chat ${get().sessions.length + 1}`,
          createdAt: now,
          usageWindowStartedAt: now,
          isActive: true,
          needsAttention: false,
          model: model ?? null,
          project: normalizeProjectName(project) ?? undefined,
          autoApprovalEnabled: false,
          autoApprovalCostCapUsd: null,
          autoApprovalEstimatedSpendUsd: 0,
          autoApprovalRemainingUsd: null,
        };
        set((state) => ({
          sessions: [...state.sessions, newSession],
          activeSessionId: id,
        }));
      },

      deleteSession: (id: string) => {
        deleteMessages(id);
        deleteBackendMessages(id);
        set((state) => {
          const newSessions = state.sessions.filter((s) => s.id !== id);
          const newActiveId =
            state.activeSessionId === id
              ? nextActiveSessionId(newSessions)
              : state.activeSessionId;
          return {
            sessions: newSessions,
            activeSessionId: newActiveId,
          };
        });
      },

      markExpired: (id: string) => {
        // Already expired or not listed: nothing changes, so skip the write.
        if (!get().sessions.some((s) => s.id === id && !s.expired)) return;
        set((state) => ({
          sessions: state.sessions.map((s) => (s.id === id ? { ...s, expired: true } : s)),
        }));
      },

      resolveDeadSession: (id: string) => {
        const { sessions, archivedSessionIds, markExpired, deleteSession } = get();
        const row = sessions.find((s) => s.id === id);
        const action = deadSessionAction(row, {
          hasClientCache: row !== undefined && hasCachedMessages(id),
          archivedSessionIds,
        });
        if (action === 'keep') markExpired(id);
        if (action === 'drop') deleteSession(id);
      },

      setArchivedSessionIds: (ids: readonly string[]) => {
        const next = new Set(ids);
        const current = get().archivedSessionIds;
        if (current && sameMembers(current, next)) return;
        set({ archivedSessionIds: next });
      },

      setServerListsMerged: (merged: boolean) => {
        if (get().serverListsMerged !== merged) set({ serverListsMerged: merged });
      },

      clearExpired: (id: string) => {
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === id ? { ...s, expired: false } : s,
          ),
        }));
      },

      mergeServerSessions: (serverSessions) => {
        set((state) => {
          const byId = new Map(state.sessions.map((s) => [s.id, s]));
          const merged = [...state.sessions];
          for (const server of serverSessions) {
            const id = server.session_id;
            if (!id) continue;
            const existing = byId.get(id);
            if (existing) {
              const auto = server.auto_approval;
              // Keep an optimistic local rename until the server reports it.
              const keepLocalTitle = existing.titleIsCustom && !server.title_is_custom;
              const updated = {
                ...existing,
                title: keepLocalTitle ? existing.title : server.title || existing.title,
                titleIsCustom: Boolean(server.title_is_custom || existing.titleIsCustom),
                isActive: server.is_active ?? existing.isActive,
                isProcessing: Boolean(server.is_processing),
                model: server.model ?? existing.model ?? null,
                // The server's folder name wins; rows it doesn't know keep theirs.
                project: normalizeProjectName(server.project) ?? existing.project,
                usageWindowStartedAt:
                  server.usage_window_started_at ?? existing.usageWindowStartedAt ?? null,
                needsAttention: Boolean(server.pending_approval?.length) || existing.needsAttention,
                expired: false,
                ...(auto
                  ? {
                      autoApprovalEnabled: Boolean(auto.enabled),
                      autoApprovalCostCapUsd: auto.cost_cap_usd ?? null,
                      autoApprovalEstimatedSpendUsd: auto.estimated_spend_usd ?? 0,
                      autoApprovalRemainingUsd: auto.remaining_usd ?? null,
                    }
                  : {}),
              };
              const idx = merged.findIndex((s) => s.id === id);
              if (idx >= 0) merged[idx] = updated;
              byId.set(id, updated);
              continue;
            }
            const newSession: SessionMeta = {
              id,
              title: server.title || `Chat ${merged.length + 1}`,
              titleIsCustom: Boolean(server.title && server.title_is_custom),
              createdAt: server.created_at || new Date().toISOString(),
              usageWindowStartedAt: server.usage_window_started_at ?? null,
              isActive: server.is_active ?? true,
              isProcessing: Boolean(server.is_processing),
              needsAttention: Boolean(server.pending_approval?.length),
              model: server.model ?? null,
              project: normalizeProjectName(server.project) ?? undefined,
              expired: false,
              autoApprovalEnabled: Boolean(server.auto_approval?.enabled),
              autoApprovalCostCapUsd: server.auto_approval?.cost_cap_usd ?? null,
              autoApprovalEstimatedSpendUsd: server.auto_approval?.estimated_spend_usd ?? 0,
              autoApprovalRemainingUsd: server.auto_approval?.remaining_usd ?? null,
            };
            merged.push(newSession);
            byId.set(id, newSession);
          }
          // Merging never opens a chat: with none open, the user picks one.
          // Opening the last row made a fresh browser open an arbitrary chat
          // (restored with a sandbox on a Mongo-backed backend).
          return { sessions: merged };
        });
      },

      updateSessionYolo: (id, policy) => {
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === id
              ? {
                  ...s,
                  autoApprovalEnabled: policy.enabled,
                  autoApprovalCostCapUsd: policy.cost_cap_usd ?? null,
                  autoApprovalEstimatedSpendUsd: policy.estimated_spend_usd ?? 0,
                  autoApprovalRemainingUsd: policy.remaining_usd ?? null,
                }
              : s,
          ),
        }));
      },

      renameSession: (oldId: string, newId: string) => {
        if (oldId === newId) return;
        moveMessages(oldId, newId);
        moveBackendMessages(oldId, newId);
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === oldId
              ? {
                  ...s,
                  id: newId,
                  expired: false,
                  // Remember the disk archive id so delete can also wipe
                  // the original session_logs/<oldId>.json — otherwise the
                  // next page load would re-hydrate it from disk and the
                  // row would reappear in the sidebar.
                  sourceArchiveId: s.sourceArchiveId || oldId,
                }
              : s,
          ),
          activeSessionId: state.activeSessionId === oldId ? newId : state.activeSessionId,
        }));
      },

      switchSession: (id: string) => {
        set((state) => ({
          activeSessionId: id,
          sessions: state.sessions.map((s) =>
            s.id === id ? { ...s, needsAttention: false } : s
          ),
        }));
      },

      setSessionActive: (id: string, isActive: boolean) => {
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === id ? { ...s, isActive } : s
          ),
        }));
      },

      setSessionProcessing: (id: string, isProcessing: boolean) => {
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === id ? { ...s, isProcessing } : s
          ),
        }));
      },

      updateSessionTitle: (id: string, title: string) => {
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === id && !s.titleIsCustom ? { ...s, title } : s
          ),
        }));
      },

      setSessionTitle: (id: string, title: string, titleIsCustom: boolean) => {
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === id ? { ...s, title, titleIsCustom } : s
          ),
        }));
      },

      updateSessionModel: (id: string, model: string | null) => {
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === id ? { ...s, model } : s
          ),
        }));
      },

      setNeedsAttention: (id: string, needs: boolean) => {
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === id ? { ...s, needsAttention: needs } : s
          ),
        }));
      },

      mergeArchivedSessions: (archived) => {
        set((state) => {
          const archivedIds = new Set(archived.map((a) => a.session_id).filter(Boolean));
          const archivedById = new Map(archived.map((a) => [a.session_id, a]));
          // Prune stale rows: an expired row that is no longer in the disk
          // list (the user or a cleanup script deleted the file) and has
          // nothing cached in this browser. Without this, stale localStorage
          // entries keep zombie-resurrecting on every reload. Live in-memory
          // sessions (expired: false) are left alone — they come from
          // /api/sessions, not /api/sessions/archived.
          // The caches are read at most once, and only if a row is about to go.
          let cachedIds: ReadonlySet<string> | null = null;
          const isCached = (id: string): boolean => {
            if (cachedIds === null) cachedIds = cachedMessageIds();
            return cachedIds.has(id);
          };
          const prunedSessions = state.sessions.filter((s) => {
            if (!s.expired) return true;            // active local row, keep
            if (archivedIds.has(s.id)) return true; // still on disk, keep
            // Also keep if its sourceArchiveId (pre-rename) is on disk —
            // covers the Catch-me-up case where the local id was renamed
            // to a fresh in-memory id but the disk file kept the original.
            if (s.sourceArchiveId && archivedIds.has(s.sourceArchiveId)) return true;
            // The disk list leaves out short chats (under 10 messages, or no
            // assistant reply), so a missing row may still be on disk. Keep
            // it while this browser caches its messages: Catch me up can
            // restore it from that cache.
            if (isCached(s.id)) return true;
            return false; // stale → drop
          }).map((s) => withArchivedProject(withArchivedCustomTitle(s, archivedById), archivedById));
          // knownIds includes BOTH the current row id AND its sourceArchiveId
          // (set when "Catch me up" renamed the row). Without this, a reload
          // after Catch me up would re-add the original archive as a NEW
          // duplicate row (since the renamed id no longer matches).
          const knownIds = new Set<string>();
          for (const s of prunedSessions) {
            knownIds.add(s.id);
            if (s.sourceArchiveId) knownIds.add(s.sourceArchiveId);
          }
          // Group by (project + first 60 chars of preview) to deduplicate
          // multiple archives of the same session (auto-saved across runs).
          // Within each group, keep only the largest message_count.
          const grouped = new Map<string, typeof archived[number]>();
          for (const a of archived) {
            if (!a.session_id || knownIds.has(a.session_id)) continue;
            const preview = (a.first_user_preview || '').replace(/\s+/g, ' ').trim().slice(0, 60);
            const proj = (a as { project?: string }).project || '';
            const key = `${proj}::${preview}`;
            const existing = grouped.get(key);
            if (!existing || ((a.message_count || 0) > (existing.message_count || 0))) {
              grouped.set(key, a);
            }
          }

          const additions: SessionMeta[] = [];
          for (const a of grouped.values()) {
            const preview = (a.first_user_preview || '').replace(/\s+/g, ' ').trim().slice(0, 80);
            const title = a.title || preview || `Archived session`;
            const proj = (a as { project?: string }).project;
            additions.push({
              id: a.session_id,
              title,
              titleIsCustom: Boolean(a.title && a.title_is_custom),
              archivePreview: preview,
              createdAt: a.saved_at || new Date().toISOString(),
              isActive: false,
              needsAttention: false,
              expired: true,
              messageCount: a.message_count,
              project: proj || undefined,
              savedAt: a.saved_at || undefined,
              sourceArchiveId: a.session_id, // identical at hydration time
            });
          }
          // Final de-dup pass over the COMBINED list (existing rows +
          // additions + any live rows from mergeServerSessions). The grouping
          // above only de-dups the new additions; without this pass, two
          // archives of the same conversation saved under different session
          // ids across runs would each survive every reload (both files are
          // on disk, so the prune keeps both) and show as duplicate rows.
          // Only rows that came from disk (they carry archivePreview, also
          // after Catch me up) collapse, on (project + normalized preview),
          // keeping the best representative. Every other row keys on its id:
          // two different chats can share a title ("Chat 3", or the same auto
          // title), and keying them on it hid one of them on every load.
          const combined = [...prunedSessions, ...additions];
          const dedupKeyOf = (s: SessionMeta): string => {
            if (s.archivePreview === undefined) return `id::${s.id}`;
            const t = s.archivePreview
              .toLowerCase()
              .replace(/\s+/g, ' ')
              .trim()
              .slice(0, 60);
            if (!t || t === 'archived session') return `id::${s.id}`;
            return `${s.project || ''}::${t}`;
          };
          // Pick the best representative of a duplicate group: a live
          // (in-memory) session always beats an archive; otherwise keep the
          // most complete (largest message_count), then the most recent.
          const isBetter = (cand: SessionMeta, cur: SessionMeta): boolean => {
            if (!!cand.expired !== !!cur.expired) return !cand.expired;
            const cm = cand.messageCount || 0;
            const um = cur.messageCount || 0;
            if (cm !== um) return cm > um;
            return (
              (cand.savedAt || cand.createdAt || '') >
              (cur.savedAt || cur.createdAt || '')
            );
          };
          const byKey = new Map<string, SessionMeta>();
          for (const s of combined) {
            const key = dedupKeyOf(s);
            const existing = byKey.get(key);
            if (!existing || isBetter(s, existing)) byKey.set(key, s);
          }
          const deduped = Array.from(byKey.values());
          // If the active/selected row got collapsed into a fuller sibling,
          // move the selection to the row we kept so the UI stays consistent.
          const dedupedIds = new Set(deduped.map((s) => s.id));
          let activeSessionId = state.activeSessionId;
          if (activeSessionId && !dedupedIds.has(activeSessionId)) {
            const activeRow = state.sessions.find((s) => s.id === activeSessionId);
            const kept = activeRow ? byKey.get(dedupKeyOf(activeRow)) : undefined;
            activeSessionId = kept ? kept.id : null;
          }
          // No-op guard: same row objects in the same order AND unchanged
          // selection (a refreshed archived title yields a new row object).
          if (
            activeSessionId === state.activeSessionId &&
            deduped.length === state.sessions.length &&
            deduped.every((s, i) => s === state.sessions[i])
          ) {
            return state;
          }
          return { ...state, sessions: deduped, activeSessionId };
        });
      },
    }),
    {
      name: 'hf-agent-sessions',
      // A full localStorage must not make actions like createSession throw.
      storage: createJSONStorage(() => safeLocalStorage),
      // Persist only these two: archivedSessionIds and serverListsMerged
      // describe this page load's server lists.
      partialize: (state) => ({
        // Reset transient processing so cold-load state is re-derived from
        // the live GET /sessions list.
        sessions: state.sessions.map((s) => ({
          ...s,
          isProcessing: false,
        })),
        activeSessionId: state.activeSessionId,
      }),
    }
  )
);

// Freeing localStorage space never evicts the cached messages of the chat on screen.
setActiveSessionSource(() => useSessionStore.getState().activeSessionId);
