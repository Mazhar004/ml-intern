import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Snackbar,
  Typography,
  CircularProgress,
  Divider,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import ChatBubbleOutlineIcon from '@mui/icons-material/ChatBubbleOutline';
import { useSessionStore } from '@/store/sessionStore';
import { useAgentStore } from '@/store/agentStore';
import { useUsageStore } from '@/store/usageStore';
import { apiFetch } from '@/utils/api';
import { safeGetItem, safeSetItem } from '@/lib/safeStorage';
import { normalizeSessionTitle, saveSessionTitle } from '@/utils/sessionTitle';
import {
  ALL_PROJECTS,
  countWaitingByGroup,
  groupSessionsByProject,
  normalizeProjectName,
  projectFolderName,
  projectGroupKey,
} from '@/utils/sessionGroups';
import type { SessionMeta } from '@/types/agent';
import ProjectFilter, { type ProjectFilterOption } from './ProjectFilter';
import ProjectGroupSection from './ProjectGroupSection';
import SessionRow from './SessionRow';
import { useCollapsedGroups, useProjectFilter } from './useProjectPrefs';

const LAST_PROJECT_STORAGE_KEY = 'ml-intern.lastProject';
const ALL_PROJECTS_LABEL = 'All projects';
const NO_PROJECT_LABEL = 'No project';

interface SessionSidebarProps {
  onClose?: () => void;
}

export default function SessionSidebar({ onClose }: SessionSidebarProps) {
  const {
    sessions,
    activeSessionId,
    createSession,
    deleteSession,
    switchSession,
    mergeArchivedSessions,
    mergeServerSessions,
    setSessionTitle,
  } =
    useSessionStore();
  // Select just these actions: subscribing to the whole agent store would
  // re-render the sidebar on every streamed agent update.
  const setPlan = useAgentStore((s) => s.setPlan);
  const clearPanel = useAgentStore((s) => s.clearPanel);
  const [isCreatingSession, setIsCreatingSession] = useState(false);
  const [capacityError, setCapacityError] = useState<string | null>(null);
  // Set once the mount-time server merges below have finished (or failed).
  const [serverListsMerged, setServerListsMerged] = useState(false);

  // -- Project groups + filter ----------------------------------------------
  // Rows keep the flat list's order (newest added first) inside each group.
  const newestFirst = useMemo(() => [...sessions].reverse(), [sessions]);
  const groups = useMemo(() => groupSessionsByProject(newestFirst), [newestFirst]);
  const hasProjects = groups.some((group) => group.project !== null);
  // 1-based position in store order: numbers the "Session NN" fallback title.
  const rowNumbers = useMemo(
    () => new Map(sessions.map((session, index) => [session.id, index + 1])),
    [sessions],
  );
  const { projectFilter, selectProjectFilter, revealGroup } = useProjectFilter(
    groups,
    serverListsMerged,
  );
  const filteredGroup =
    projectFilter === ALL_PROJECTS ? undefined : groups.find((g) => g.key === projectFilter);
  // Chats waiting for a tool approval, per group: a collapsed or filtered-out
  // group hides their rows' dots, so its header and filter option show one.
  const waitingByGroup = useMemo(
    () => countWaitingByGroup(groups, activeSessionId),
    [groups, activeSessionId],
  );
  const filterOptions = useMemo<ProjectFilterOption[]>(
    () => [
      { value: ALL_PROJECTS, label: ALL_PROJECTS_LABEL, count: sessions.length, waiting: 0 },
      ...groups.map((group) => ({
        value: group.key,
        label: group.project ?? NO_PROJECT_LABEL,
        count: group.count,
        waiting: waitingByGroup.get(group.key) ?? 0,
      })),
    ],
    [groups, sessions.length, waitingByGroup],
  );
  const activeGroupKey = useMemo(() => {
    const active = sessions.find((session) => session.id === activeSessionId);
    return active ? projectGroupKey(normalizeProjectName(active.project)) : null;
  }, [sessions, activeSessionId]);
  const { collapsed, toggleGroup } = useCollapsedGroups(activeSessionId, activeGroupKey);
  // New Task pre-fills the filtered project ('' for "No project"); without a
  // filter it falls back to the last-used project.
  const prefillProject = filteredGroup ? (filteredGroup.project ?? '') : null;

  // A filter that would hide a just-created chat switches to its project.
  const revealInFilter = useCallback(
    (project: string) => revealGroup(projectGroupKey(normalizeProjectName(project))),
    [revealGroup],
  );

  // -- Hydrate the sidebar on mount ----------------------------------------
  // Two sources, merged: (1) /api/sessions — live/durable sessions the
  // backend currently knows about (shown as active, clickable rows);
  // (2) /api/sessions/archived — every session_logs/<uuid>_*.json on disk,
  // including ones from another browser, created via curl, or saved before
  // this browser's localStorage was populated. Archived rows show as
  // "needs a catch-up"; clicking one + Catch me up runs /api/session/import
  // for a full restore. Server sessions are merged first so the archived
  // pass de-dupes against them and never adds a duplicate live row.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const response = await apiFetch('/api/sessions');
        if (response.ok) {
          const live = await response.json();
          if (!cancelled && Array.isArray(live)) mergeServerSessions(live);
        }
      } catch {
        /* live listing is best-effort; archived + local metadata still work */
      }
      try {
        const res = await apiFetch('/api/sessions/archived');
        if (!res.ok) return;
        const data = await res.json();
        if (cancelled || !Array.isArray(data)) return;
        mergeArchivedSessions(data);
      } catch {
        // best-effort: archived listing is a convenience, not load-bearing
      }
    })().finally(() => {
      if (!cancelled) setServerListsMerged(true);
    });
    return () => {
      cancelled = true;
    };
  }, [mergeArchivedSessions, mergeServerSessions]);

  // -- Handlers -----------------------------------------------------------

  const handleNewSession = useCallback(async () => {
    if (isCreatingSession) return;
    setIsCreatingSession(true);
    setCapacityError(null);
    // Set once the backend has created the session: after that, a failing
    // local step must not be reported as "Failed to create session.".
    let created: { id: string; model?: string | null; project: string } | null = null;
    try {
      // Project picker: prompt for the project folder this session lives in.
      // Empty input → no project sandbox (server-wide ML_INTERN_PROJECT_ROOT
      // env var still applies if set). Pre-fill the filtered project, else
      // the last-used one from localStorage, so jumping between sessions on
      // the same project is fast.
      let existing: { name: string }[] = [];
      try {
        const r = await apiFetch('/api/projects');
        if (r.ok) existing = await r.json();
      } catch { /* listing is best-effort */ }
      const last = prefillProject ?? safeGetItem(LAST_PROJECT_STORAGE_KEY) ?? '';
      const hint = existing.length
        ? `\n\nExisting projects:\n  ${existing.map((p) => p.name).join('\n  ')}`
        : '';
      const project = window.prompt(
        `Project name for this session (leave blank for none).\n` +
        `Files written by the agent will go into projects/<name>/.${hint}`,
        last,
      );
      // null means user cancelled the prompt — abort entirely.
      if (project === null) {
        return;
      }
      const trimmed = project.trim();
      // The backend files the chat under a cleaned-up folder name ("a/b" is
      // folder "a-b"); group, filter and pre-fill with that same name.
      const folder = projectFolderName(trimmed);
      if (folder) {
        safeSetItem(LAST_PROJECT_STORAGE_KEY, folder);
      }

      const response = await apiFetch('/api/session', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(trimmed ? { project: trimmed } : {}),
      });
      if (response.status === 503) {
        const data = await response.json();
        setCapacityError(data.detail || 'Server is at capacity.');
        return;
      }
      if (!response.ok) {
        throw new Error(`POST /api/session returned ${response.status}`);
      }
      const data = await response.json();
      if (typeof data.session_id !== 'string' || !data.session_id) {
        throw new Error('POST /api/session returned no session_id');
      }
      created = { id: data.session_id, model: data.model, project: folder };
      createSession(data.session_id, data.model, folder);
      revealInFilter(folder);
      void useUsageStore.getState().fetchUsage(data.session_id);
      setPlan([]);
      clearPanel();
      onClose?.();
    } catch (error) {
      if (created) {
        const { id, model, project } = created;
        console.error(`Session ${id} was created, but setting it up locally failed:`, error);
        // Best effort: the backend session exists, so still put it on screen.
        try {
          if (useSessionStore.getState().sessions.some((s) => s.id === id)) switchSession(id);
          else createSession(id, model, project);
          revealInFilter(project);
          onClose?.();
        } catch (switchError) {
          console.error(`Could not switch to new session ${id}:`, switchError);
        }
      } else {
        console.error('Failed to create session:', error);
        setCapacityError('Failed to create session.');
      }
    } finally {
      setIsCreatingSession(false);
    }
  }, [
    isCreatingSession,
    prefillProject,
    createSession,
    revealInFilter,
    switchSession,
    setPlan,
    clearPanel,
    onClose,
  ]);

  // -- Delete with dialog confirmation ------------------------------------
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  const handleDeleteClick = useCallback(
    (sessionId: string, e: React.MouseEvent) => {
      e.stopPropagation();
      setConfirmDeleteId(sessionId);
    },
    [],
  );

  const handleDeleteConfirm = useCallback(async () => {
    if (!confirmDeleteId || isDeleting) return;
    const sessionId = confirmDeleteId;
    const sessionRow = sessions.find((s) => s.id === sessionId);
    setIsDeleting(true);

    const isLastSession = sessions.length === 1;

    useAgentStore.getState().clearSessionState(sessionId);
    try {
      // Wipe ALL layers so the row doesn't reappear on next reload:
      //   1. In-memory backend session (if any)
      //   2. Disk archive at the CURRENT id
      //   3. Disk archive at the ORIGINAL id (before Catch me up renamed it)
      const archiveIds = new Set<string>([sessionId]);
      if (sessionRow?.sourceArchiveId) archiveIds.add(sessionRow.sourceArchiveId);
      await Promise.allSettled([
        apiFetch(`/api/session/${sessionId}`, { method: 'DELETE' }),
        ...Array.from(archiveIds).map((id) =>
          apiFetch(`/api/sessions/archived/${id}`, { method: 'DELETE' }),
        ),
      ]);
    } catch {
      /* best effort: the local row is removed below either way */
    }
    // Deleting the open chat makes the store open the newest chat left, which
    // a project filter may hide: open the filtered project's newest instead.
    const { sessions: current, activeSessionId: openId } = useSessionStore.getState();
    if (openId === sessionId && projectFilter !== ALL_PROJECTS) {
      const next = [...current]
        .reverse()
        .find(
          (s) =>
            s.id !== sessionId &&
            projectGroupKey(normalizeProjectName(s.project)) === projectFilter,
        );
      if (next) switchSession(next.id);
    }
    deleteSession(sessionId);

    // If this was the last session, create a new one
    if (isLastSession) {
      try {
        const response = await apiFetch('/api/session', { method: 'POST' });
        if (response.ok) {
          const data = await response.json();
          createSession(data.session_id, data.model);
          void useUsageStore.getState().fetchUsage(data.session_id);
          setPlan([]);
          clearPanel();
        }
      } catch (error) {
        console.error('Failed to create new session after deleting last one:', error);
      }
    }

    setIsDeleting(false);
    setConfirmDeleteId(null);
  }, [
    deleteSession,
    confirmDeleteId,
    isDeleting,
    sessions,
    projectFilter,
    switchSession,
    createSession,
    setPlan,
    clearPanel,
  ]);

  const handleSelect = useCallback(
    (sessionId: string) => {
      const shouldActivateSession = sessionId !== activeSessionId;
      switchSession(sessionId);
      if (shouldActivateSession) {
        void (async () => {
          try {
            const response = await apiFetch(`/api/session/${sessionId}/activate`, {
              method: 'POST',
            });
            if (!response.ok) return;
            const info = await response.json();
            mergeServerSessions([info]);
            void useUsageStore.getState().fetchUsage(sessionId);
          } catch {
            /* best effort: usage falls back to the existing session metadata */
          }
        })();
      }
      // Per-session state (plan, panel, activity) is restored automatically
      // by SessionChat's useEffect when isActive flips to true.
      onClose?.();
    },
    [activeSessionId, mergeServerSessions, switchSession, onClose],
  );

  // -- Rename (inline editor) ---------------------------------------------
  const [editingId, setEditingId] = useState<string | null>(null);
  const [renameError, setRenameError] = useState<string | null>(null);

  const startRename = useCallback((sessionId: string, e: React.SyntheticEvent) => {
    e.stopPropagation();
    setEditingId(sessionId);
  }, []);

  const cancelRename = useCallback(() => setEditingId(null), []);

  // Optimistic: show the new name at once, revert if the server rejects it.
  const commitRename = useCallback(
    async (session: SessionMeta, raw: string, shownTitle: string) => {
      setEditingId(null);
      const title = normalizeSessionTitle(raw);
      if (!title || title === shownTitle) return;
      setSessionTitle(session.id, title, true);
      try {
        const saved = await saveSessionTitle(session, title);
        if (saved !== title) setSessionTitle(session.id, saved, true);
      } catch (error) {
        const current = useSessionStore.getState().sessions.find((s) => s.id === session.id);
        if (current?.title === title) {
          setSessionTitle(session.id, session.title, Boolean(session.titleIsCustom));
        }
        setRenameError(error instanceof Error ? error.message : 'Could not rename chat.');
      }
    },
    [setSessionTitle],
  );

  // -- Render -------------------------------------------------------------

  const renderRow = (session: SessionMeta) => (
    <SessionRow
      key={session.id}
      session={session}
      num={rowNumbers.get(session.id) ?? 0}
      isSelected={session.id === activeSessionId}
      isEditing={session.id === editingId}
      onSelect={handleSelect}
      onStartRename={startRename}
      onCommitRename={commitRename}
      onCancelRename={cancelRename}
      onDeleteClick={handleDeleteClick}
    />
  );

  // Flat list until some chat has a project; then one collapsible section
  // per project, or only the filtered project's chats.
  const renderSessionList = () => {
    if (!hasProjects) return newestFirst.map(renderRow);
    if (filteredGroup) return filteredGroup.sessions.map(renderRow);
    return groups.map((group) => {
      const expanded = !collapsed.has(group.key);
      return (
        <ProjectGroupSection
          key={group.key}
          groupKey={group.key}
          label={group.project ?? NO_PROJECT_LABEL}
          count={group.count}
          waiting={waitingByGroup.get(group.key) ?? 0}
          isProject={group.project !== null}
          expanded={expanded}
          onToggle={toggleGroup}
        >
          {expanded && group.sessions.map(renderRow)}
        </ProjectGroupSection>
      );
    });
  };

  return (
    <Box
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        bgcolor: 'var(--panel)',
      }}
    >
      {/* -- Header -------------------------------------------------------- */}
      <Box sx={{ px: 1.75, pt: 2, pb: 0 }}>
        <Typography
          variant="caption"
          sx={{
            color: 'var(--muted-text)',
            fontSize: '0.65rem',
            fontWeight: 600,
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
          }}
        >
          Recent chats
        </Typography>
      </Box>

      {/* -- Project filter (once any chat has a project) ------------------ */}
      {hasProjects && (
        <Box sx={{ px: 1.5, pt: 1 }}>
          <ProjectFilter
            value={projectFilter}
            options={filterOptions}
            onChange={selectProjectFilter}
          />
        </Box>
      )}

      {/* -- Capacity error ------------------------------------------------ */}
      {capacityError && (
        <Alert
          severity="warning"
          variant="outlined"
          onClose={() => setCapacityError(null)}
          sx={{
            m: 1,
            fontSize: '0.7rem',
            py: 0.25,
            '& .MuiAlert-message': { py: 0 },
            borderColor: '#FF9D00',
            color: 'var(--text)',
          }}
        >
          {capacityError}
        </Alert>
      )}

      {/* -- Session list -------------------------------------------------- */}
      <Box
        sx={{
          flex: 1,
          overflow: 'auto',
          py: 1,
          '&::-webkit-scrollbar': { width: 4 },
          '&::-webkit-scrollbar-thumb': {
            bgcolor: 'var(--scrollbar-thumb)',
            borderRadius: 2,
          },
        }}
      >
        {sessions.length === 0 ? (
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              py: 8,
              px: 3,
              gap: 1.5,
            }}
          >
            <ChatBubbleOutlineIcon
              sx={{ fontSize: 28, color: 'var(--muted-text)', opacity: 0.25 }}
            />
            <Typography
              variant="caption"
              sx={{
                color: 'var(--muted-text)',
                opacity: 0.5,
                textAlign: 'center',
                lineHeight: 1.5,
                fontSize: '0.72rem',
              }}
            >
              No sessions yet
            </Typography>
          </Box>
        ) : (
          renderSessionList()
        )}
      </Box>

      {/* -- Footer: New Task + status ------------------------------------- */}
      <Divider sx={{ opacity: 0.5 }} />
      <Box
        sx={{
          px: 1.5,
          py: 1.5,
          display: 'flex',
          flexDirection: 'column',
          gap: 1,
          flexShrink: 0,
        }}
      >
        <Box
          component="button"
          onClick={handleNewSession}
          disabled={isCreatingSession}
          sx={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 0.75,
            width: '100%',
            px: 1.5,
            py: 1.25,
            border: 'none',
            borderRadius: '10px',
            bgcolor: '#FF9D00',
            color: '#000',
            fontSize: '0.85rem',
            fontWeight: 700,
            cursor: 'pointer',
            transition: 'all 0.12s ease',
            '&:hover': {
              bgcolor: '#FFB340',
            },
            '&:disabled': {
              opacity: 0.5,
              cursor: 'not-allowed',
            },
          }}
        >
          {isCreatingSession ? (
            <>
              <CircularProgress size={12} sx={{ color: '#000' }} />
              Creating...
            </>
          ) : (
            <>
              <AddIcon sx={{ fontSize: 16 }} />
              New Task
            </>
          )}
        </Box>

      </Box>
      <Snackbar
        open={!!renameError}
        autoHideDuration={4000}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        onClose={() => setRenameError(null)}
      >
        <Alert
          severity="error"
          variant="filled"
          onClose={() => setRenameError(null)}
          sx={{ fontSize: '0.8rem' }}
        >
          {renameError}
        </Alert>
      </Snackbar>
      {/* Delete confirmation dialog */}
      <Dialog
        open={!!confirmDeleteId}
        onClose={() => !isDeleting && setConfirmDeleteId(null)}
        slotProps={{
          backdrop: { sx: { backgroundColor: 'rgba(0,0,0,0.5)', backdropFilter: 'blur(4px)' } },
        }}
        PaperProps={{
          sx: {
            bgcolor: 'var(--panel)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)',
            boxShadow: 'var(--shadow-1)',
            maxWidth: 340,
            mx: 2,
          },
        }}
      >
        <DialogTitle
          sx={{
            color: 'var(--text)',
            fontWeight: 700,
            fontSize: '0.95rem',
            pb: 0,
            pt: 2.5,
            px: 3,
          }}
        >
          Delete conversation?
        </DialogTitle>
        <DialogContent sx={{ px: 3, pt: 1 }}>
          <DialogContentText
            sx={{
              color: 'var(--muted-text)',
              fontSize: '0.82rem',
              lineHeight: 1.6,
            }}
          >
            This will permanently remove this conversation and its history.
          </DialogContentText>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2.5, gap: 1 }}>
          <Button
            onClick={() => setConfirmDeleteId(null)}
            size="small"
            disabled={isDeleting}
            sx={{
              color: 'var(--muted-text)',
              fontSize: '0.82rem',
              px: 2,
              '&:hover': { bgcolor: 'var(--hover-bg)' },
            }}
          >
            Cancel
          </Button>
          <Button
            onClick={handleDeleteConfirm}
            variant="contained"
            size="small"
            disabled={isDeleting}
            startIcon={isDeleting ? <CircularProgress size={16} sx={{ color: '#fff' }} /> : undefined}
            sx={{
              fontSize: '0.82rem',
              px: 2.5,
              bgcolor: 'var(--accent-red)',
              color: '#fff',
              boxShadow: 'none',
              '&:hover': {
                bgcolor: 'var(--accent-red)',
                filter: 'brightness(1.15)',
                boxShadow: 'none',
              },
              '&.Mui-disabled': {
                bgcolor: 'var(--accent-red)',
                color: '#fff',
                opacity: 0.7,
              },
            }}
          >
            {isDeleting ? 'Deleting...' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
