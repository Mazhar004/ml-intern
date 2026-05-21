import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { SessionMeta } from '@/types/agent';
import { deleteMessages, moveMessages } from '@/lib/chat-message-store';
import { moveBackendMessages, deleteBackendMessages } from '@/lib/backend-message-store';

interface SessionStore {
  sessions: SessionMeta[];
  activeSessionId: string | null;

  // Actions
  createSession: (id: string) => void;
  deleteSession: (id: string) => void;
  switchSession: (id: string) => void;
  setSessionActive: (id: string, isActive: boolean) => void;
  updateSessionTitle: (id: string, title: string) => void;
  setNeedsAttention: (id: string, needs: boolean) => void;
  /** Mark a session as expired (backend no longer has it). The UI shows a
   *  recovery banner and disables input. */
  markExpired: (id: string) => void;
  /** Clear the expired flag (used after restore-with-summary succeeds). */
  clearExpired: (id: string) => void;
  /** Atomically swap a session's id in the list + both localStorage caches.
   *  Used when we rehydrate an expired session into a freshly-created backend
   *  session — preserves title, timestamps, and messages. */
  renameSession: (oldId: string, newId: string) => void;
  /** Merge a list of archived (server-side, on-disk) sessions into the
   *  sidebar so they're visible & clickable. Skips ids already in the store
   *  and marks newly-added ones as expired so the UI shows the recovery
   *  banner — clicking "Catch me up" then triggers a full /api/session/import
   *  rehydration. Items: {session_id, saved_at, message_count, first_user_preview}. */
  mergeArchivedSessions: (
    archived: Array<{ session_id: string; saved_at?: string | null; message_count?: number; first_user_preview?: string }>,
  ) => void;
}

export const useSessionStore = create<SessionStore>()(
  persist(
    (set, get) => ({
      sessions: [],
      activeSessionId: null,

      createSession: (id: string) => {
        const newSession: SessionMeta = {
          id,
          title: `Chat ${get().sessions.length + 1}`,
          createdAt: new Date().toISOString(),
          isActive: true,
          needsAttention: false,
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
              ? newSessions[newSessions.length - 1]?.id || null
              : state.activeSessionId;
          return {
            sessions: newSessions,
            activeSessionId: newActiveId,
          };
        });
      },

      markExpired: (id: string) => {
        set((state) => ({
          sessions: state.sessions.map((s) => (s.id === id ? { ...s, expired: true } : s)),
        }));
      },

      clearExpired: (id: string) => {
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === id ? { ...s, expired: false } : s,
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

      updateSessionTitle: (id: string, title: string) => {
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === id ? { ...s, title } : s
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
          // Prune stale rows: any row that was previously an archived/expired
          // entry but is no longer in the disk list — that means the user
          // (or a cleanup script) deleted the file. Without this, stale
          // localStorage entries keep zombie-resurrecting on every reload.
          // Live in-memory sessions (expired: false) are left alone — they
          // come from /api/sessions, not /api/sessions/archived.
          const prunedSessions = state.sessions.filter((s) => {
            if (!s.expired) return true;            // active local row, keep
            if (archivedIds.has(s.id)) return true; // still on disk, keep
            // Also keep if its sourceArchiveId (pre-rename) is on disk —
            // covers the Catch-me-up case where the local id was renamed
            // to a fresh in-memory id but the disk file kept the original.
            if (s.sourceArchiveId && archivedIds.has(s.sourceArchiveId)) return true;
            return false; // stale → drop
          });
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
            const preview = (a.first_user_preview || '').replace(/\s+/g, ' ').trim();
            const title = preview ? preview.slice(0, 80) : `Archived session`;
            const proj = (a as { project?: string }).project;
            additions.push({
              id: a.session_id,
              title,
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
          // If nothing pruned and nothing added, no change.
          if (additions.length === 0 && prunedSessions.length === state.sessions.length) {
            return state;
          }
          return { ...state, sessions: [...prunedSessions, ...additions] };
        });
      },
    }),
    {
      name: 'hf-agent-sessions',
      partialize: (state) => ({
        sessions: state.sessions,
        activeSessionId: state.activeSessionId,
      }),
    }
  )
);
