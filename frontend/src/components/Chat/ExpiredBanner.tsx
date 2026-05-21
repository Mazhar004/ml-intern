/**
 * Shown inline in a chat when the backend no longer recognizes the
 * session id (typically: Space was restarted). Lets the user catch the
 * agent up with a summary of the prior conversation, or start over.
 */
import { useState, useCallback } from 'react';
import { Box, Button, CircularProgress, Typography } from '@mui/material';
import { apiFetch } from '@/utils/api';
import { useSessionStore } from '@/store/sessionStore';
import { useAgentStore } from '@/store/agentStore';
import { loadBackendMessages } from '@/lib/backend-message-store';
import { loadMessages } from '@/lib/chat-message-store';
import { uiMessagesToLLMMessages } from '@/lib/convert-llm-messages';
import { logger } from '@/utils/logger';

interface Props {
  sessionId: string;
}

export default function ExpiredBanner({ sessionId }: Props) {
  const { renameSession, deleteSession, updateSessionModel } = useSessionStore();
  const [busy, setBusy] = useState<'catch-up' | 'start-over' | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleCatchUp = useCallback(async () => {
    setBusy('catch-up');
    setError(null);
    try {
      // Strategy 1 (preferred): full restore from session_logs/ on disk via
      // /api/session/import. Loses nothing — every original message, every
      // tool call comes back.
      const importRes = await apiFetch('/api/session/import', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      });
      if (importRes.ok) {
        const data = await importRes.json();
        const newId = data.session_id as string | undefined;
        if (newId) {
          useAgentStore.getState().clearSessionState(sessionId);
          renameSession(sessionId, newId);
          return;
        }
      }

      // Strategy 2 (fallback, lossy): if no archive on disk, rebuild from
      // the browser's localStorage cache and summarize. Older flow.
      let messages = loadBackendMessages(sessionId);
      if (!messages || messages.length === 0) {
        const uiMsgs = loadMessages(sessionId);
        if (uiMsgs.length > 0) messages = uiMessagesToLLMMessages(uiMsgs);
      }
      if (!messages || messages.length === 0) {
        setError("This session isn't on disk and your browser cache is empty — nothing to restore. Click Start fresh.");
        setBusy(null);
        return;
      }

      const res = await apiFetch('/api/session/restore-summary', {
        method: 'POST',
        body: JSON.stringify({ messages }),
      });
      if (!res.ok) throw new Error(`restore-summary failed: ${res.status}`);
      const data = await res.json();
      const newId = data.session_id as string | undefined;
      if (!newId) throw new Error('no session_id in response');

      useAgentStore.getState().clearSessionState(sessionId);
      renameSession(sessionId, newId);
      if (data.model) updateSessionModel(newId, data.model);
    } catch (e) {
      logger.warn('Catch-up failed:', e);
      setError("Couldn't catch up — try starting over.");
      setBusy(null);
    }
  }, [sessionId, renameSession, updateSessionModel]);

  const handleStartOver = useCallback(() => {
    setBusy('start-over');
    useAgentStore.getState().clearSessionState(sessionId);
    deleteSession(sessionId);
  }, [sessionId, deleteSession]);

  return (
    <Box
      sx={{
        mx: { xs: 2, md: 'auto' },
        my: 2,
        maxWidth: 720,
        p: 2.5,
        borderRadius: 2,
        border: '1px solid',
        borderColor: 'divider',
        bgcolor: 'background.paper',
        boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
      }}
    >
      <Typography variant="body1" sx={{ fontWeight: 600, mb: 0.5 }}>
        Where were we?
      </Typography>
      <Typography variant="body2" sx={{ color: 'text.secondary', mb: 2 }}>
        Let me skim the conversation so far and pick up right where we left
        off — or we can start something new.
      </Typography>
      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
        <Button
          variant="contained"
          onClick={handleCatchUp}
          disabled={busy !== null}
          startIcon={busy === 'catch-up' ? <CircularProgress size={16} color="inherit" /> : null}
          sx={{ textTransform: 'none' }}
        >
          {busy === 'catch-up' ? 'Catching up…' : 'Catch me up'}
        </Button>
        <Button
          variant="outlined"
          onClick={handleStartOver}
          disabled={busy !== null}
          sx={{ textTransform: 'none' }}
        >
          Start fresh
        </Button>
      </Box>
      {error && (
        <Typography variant="caption" sx={{ display: 'block', mt: 1.5, color: 'error.main' }}>
          {error}
        </Typography>
      )}
    </Box>
  );
}
