import { useCallback, useEffect, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  IconButton,
  Typography,
  CircularProgress,
  Divider,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import ChatBubbleOutlineIcon from '@mui/icons-material/ChatBubbleOutline';
import { useSessionStore } from '@/store/sessionStore';
import { useAgentStore } from '@/store/agentStore';
import { useUsageStore } from '@/store/usageStore';
import { apiFetch } from '@/utils/api';

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
  } =
    useSessionStore();
  const { setPlan, clearPanel } =
    useAgentStore();
  const [isCreatingSession, setIsCreatingSession] = useState(false);
  const [capacityError, setCapacityError] = useState<string | null>(null);

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
    })();
    return () => {
      cancelled = true;
    };
  }, [mergeArchivedSessions, mergeServerSessions]);

  // -- Handlers -----------------------------------------------------------

  const handleNewSession = useCallback(async () => {
    if (isCreatingSession) return;
    setIsCreatingSession(true);
    setCapacityError(null);
    try {
      // Project picker: prompt for the project folder this session lives in.
      // Empty input → no project sandbox (server-wide ML_INTERN_PROJECT_ROOT
      // env var still applies if set). Pre-fill the last-used project from
      // localStorage so jumping between sessions on the same project is fast.
      let existing: { name: string }[] = [];
      try {
        const r = await apiFetch('/api/projects');
        if (r.ok) existing = await r.json();
      } catch { /* listing is best-effort */ }
      const last = localStorage.getItem('ml-intern.lastProject') || '';
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
      if (trimmed) {
        localStorage.setItem('ml-intern.lastProject', trimmed);
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
      const data = await response.json();
      createSession(data.session_id, data.model);
      void useUsageStore.getState().fetchUsage(data.session_id);
      setPlan([]);
      clearPanel();
      onClose?.();
    } catch {
      setCapacityError('Failed to create session.');
    } finally {
      setIsCreatingSession(false);
    }
  }, [isCreatingSession, createSession, setPlan, clearPanel, onClose]);

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
      deleteSession(sessionId);
    } catch {
      deleteSession(sessionId);
    }

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
  }, [deleteSession, confirmDeleteId, isDeleting, sessions, createSession, setPlan, clearPanel]);

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

  const formatTime = (d: string) =>
    new Date(d).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  // -- Render -------------------------------------------------------------

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
          [...sessions].reverse().map((session, index) => {
            const num = sessions.length - index;
            const isSelected = session.id === activeSessionId;

            return (
              <Box
                key={session.id}
                onClick={() => handleSelect(session.id)}
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1,
                  px: 1.5,
                  py: 0.875,
                  mx: 0.75,
                  mb: 0.2,
                  borderRadius: '10px',
                  cursor: 'pointer',
                  transition: 'background-color 0.12s ease',
                  bgcolor: isSelected
                    ? 'var(--hover-bg)'
                    : 'transparent',
                  '&:hover': {
                    bgcolor: 'var(--hover-bg)',
                  },
                  '& .delete-btn': {
                    opacity: 0,
                    transition: 'opacity 0.12s',
                  },
                  '&:hover .delete-btn': {
                    opacity: 1,
                  },
                }}
              >
                <ChatBubbleOutlineIcon
                  sx={{
                    fontSize: 15,
                    color: isSelected ? 'var(--text)' : 'var(--muted-text)',
                    opacity: isSelected ? 0.8 : 0.4,
                    flexShrink: 0,
                  }}
                />

                <Box sx={{ flex: 1, minWidth: 0 }}>
                  {/* Title row: project tag + truncated preview */}
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, minWidth: 0 }}>
                    {session.project && (
                      <Box
                        component="span"
                        sx={{
                          fontSize: '0.6rem',
                          fontWeight: 600,
                          color: 'var(--accent-yellow)',
                          bgcolor: 'rgba(255, 168, 0, 0.12)',
                          px: 0.6,
                          py: 0.05,
                          borderRadius: 0.5,
                          flexShrink: 0,
                          textTransform: 'uppercase',
                          letterSpacing: 0.3,
                          maxWidth: 100,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {session.project}
                      </Box>
                    )}
                    <Typography
                      variant="body2"
                      sx={{
                        fontWeight: isSelected ? 600 : 400,
                        color: 'var(--text)',
                        fontSize: '0.84rem',
                        lineHeight: 1.4,
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        flex: 1,
                        minWidth: 0,
                      }}
                    >
                      {session.title.startsWith('Chat ') ? `Session ${String(num).padStart(2, '0')}` : session.title}
                    </Typography>
                  </Box>
                  {/* Subtitle row: status · message count · time */}
                  <Typography
                    variant="caption"
                    sx={{
                      color: 'var(--muted-text)',
                      fontSize: '0.65rem',
                      lineHeight: 1.2,
                    }}
                  >
                    {[
                      session.expired ? 'needs a catch-up' : null,
                      session.messageCount ? `${session.messageCount} msgs` : null,
                      formatTime(session.savedAt || session.createdAt),
                    ].filter(Boolean).join(' · ')}
                  </Typography>
                </Box>

                {/* Attention badge — pulsing dot when background session needs approval */}
                {session.needsAttention && !isSelected && (
                  <Box
                    sx={{
                      width: 8,
                      height: 8,
                      borderRadius: '50%',
                      bgcolor: 'var(--accent-yellow)',
                      flexShrink: 0,
                      animation: 'pulse 2s ease-in-out infinite',
                      '@keyframes pulse': {
                        '0%, 100%': { opacity: 1, transform: 'scale(1)' },
                        '50%': { opacity: 0.5, transform: 'scale(0.8)' },
                      },
                    }}
                  />
                )}

                <IconButton
                  className="delete-btn"
                  size="small"
                  onClick={(e) => handleDeleteClick(session.id, e)}
                  sx={{
                    color: 'var(--muted-text)',
                    width: 26,
                    height: 26,
                    flexShrink: 0,
                    '&:hover': { color: 'var(--accent-red)', bgcolor: 'rgba(244,67,54,0.08)' },
                  }}
                >
                  <DeleteOutlineIcon sx={{ fontSize: 15 }} />
                </IconButton>
              </Box>
            );
          })
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
