import { useCallback, useRef, useEffect, useState } from 'react';
import {
  Avatar,
  Box,
  CircularProgress,
  Drawer,
  Fade,
  Typography,
  IconButton,
  Alert,
  AlertTitle,
  Button,
  Snackbar,
  useMediaQuery,
  useTheme,
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import DragIndicatorIcon from '@mui/icons-material/DragIndicator';
import DarkModeOutlinedIcon from '@mui/icons-material/DarkModeOutlined';
import LightModeOutlinedIcon from '@mui/icons-material/LightModeOutlined';

import { useSessionStore } from '@/store/sessionStore';
import { useAgentStore } from '@/store/agentStore';
import { useLayoutStore } from '@/store/layoutStore';
import SessionSidebar from '@/components/SessionSidebar/SessionSidebar';
import SessionChat from '@/components/SessionChat';
import CodePanel from '@/components/CodePanel/CodePanel';
import WelcomeScreen from '@/components/WelcomeScreen/WelcomeScreen';
import YoloControl from '@/components/YoloControl';
import UsageMeter from '@/components/UsageMeter';
import { useServerSessionHydration } from '@/hooks/useServerSessionHydration';
import { apiFetch } from '@/utils/api';
import {
  creditCtaForModel,
  inferenceCreditCta,
  isInferenceCreditError,
  usesProviderNotices,
} from '@/utils/inferenceBilling';

const DRAWER_WIDTH = 260;

export default function AppLayout() {
  // resolveDeadSession: the backend lost a chat (404). The store keeps it for
  // Catch me up, or drops it when it is a fresh live chat with nothing to restore.
  const { sessions, activeSessionId, resolveDeadSession } = useSessionStore();
  // Loaded here, not in the sidebar: this component is always mounted.
  const { initialLoadPending } = useServerSessionHydration();
  const { isConnected, llmHealthError, setLlmHealthError, user } = useAgentStore();
  const {
    isLeftSidebarOpen,
    isRightPanelOpen,
    rightPanelWidth,
    themeMode,
    setRightPanelWidth,
    setLeftSidebarOpen,
    toggleLeftSidebar,
    toggleTheme,
  } = useLayoutStore();

  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  const [showExpiredToast, setShowExpiredToast] = useState(false);
  const disconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const isResizing = useRef(false);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isResizing.current) return;
    const newWidth = window.innerWidth - e.clientX;
    const maxWidth = window.innerWidth * 0.6;
    const minWidth = 300;
    if (newWidth > minWidth && newWidth < maxWidth) {
      setRightPanelWidth(newWidth);
    }
  }, [setRightPanelWidth]);

  const stopResizing = useCallback(() => {
    isResizing.current = false;
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', stopResizing);
    document.body.style.cursor = 'default';
  }, [handleMouseMove]);

  const startResizing = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isResizing.current = true;
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', stopResizing);
    document.body.style.cursor = 'col-resize';
  }, [handleMouseMove, stopResizing]);

  useEffect(() => {
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', stopResizing);
    };
  }, [handleMouseMove, stopResizing]);

  // -- LLM health check on mount -----------------------------------------
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await apiFetch('/api/health/llm');
        const data = await res.json();
        if (!cancelled && data.status === 'error') {
          setLlmHealthError({
            error: data.error || 'Unknown LLM error',
            errorType: data.error_type || 'unknown',
            model: data.model,
          });
        } else if (!cancelled) {
          setLlmHealthError(null);
        }
      } catch {
        // Backend unreachable -- not an LLM issue, ignore
      }
    })();
    return () => { cancelled = true; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const hasAnySessions = sessions.length > 0;

  // Debounced "session expired" toast
  useEffect(() => {
    if (!isConnected && activeSessionId) {
      disconnectTimer.current = setTimeout(() => setShowExpiredToast(true), 2000);
    } else {
      if (disconnectTimer.current) clearTimeout(disconnectTimer.current);
      disconnectTimer.current = null;
      setShowExpiredToast(false);
    }
    return () => {
      if (disconnectTimer.current) clearTimeout(disconnectTimer.current);
    };
  }, [isConnected, activeSessionId]);

  // Best-effort sandbox cleanup when the browser tab/window closes. This
  // preserves durable chat history; explicit delete still removes the session.
  useEffect(() => {
    const teardownSandboxes = () => {
      const liveSessionIds = useSessionStore
        .getState()
        .sessions
        .filter((session) => session.isActive && !session.expired)
        .map((session) => session.id);

      for (const sessionId of liveSessionIds) {
        const url = `/api/session/${sessionId}/sandbox/teardown`;
        const body = '{}';
        const blob = new Blob([body], { type: 'application/json' });

        if (navigator.sendBeacon?.(url, blob)) {
          continue;
        }

        fetch(url, {
          method: 'POST',
          body,
          keepalive: true,
          credentials: 'include',
          headers: { 'Content-Type': 'application/json' },
        }).catch(() => {});
      }
    };

    window.addEventListener('pagehide', teardownSandboxes);
    return () => window.removeEventListener('pagehide', teardownSandboxes);
  }, []);

  // Close sidebar on mobile after selecting a session
  const handleSidebarClose = useCallback(() => {
    if (isMobile) setLeftSidebarOpen(false);
  }, [isMobile, setLeftSidebarOpen]);

  // -- LLM error toast helper --------------------------------------------
  const llmErrorTitle = llmHealthError
    ? llmHealthError.errorType === 'credits'
      ? 'API Credits Exhausted'
      : llmHealthError.errorType === 'auth'
      ? 'Invalid API Key'
      : llmHealthError.errorType === 'rate_limit'
      ? 'Rate Limited'
      : llmHealthError.errorType === 'network'
      ? 'LLM Provider Unreachable'
      : 'LLM Error'
    : '';
  // The health probe targets the web default: Claude on the direct Anthropic
  // API when ANTHROPIC_API_KEY is set, or a direct OpenAI model when
  // ML_INTERN_DEFAULT_MODEL_ID names one. Their credit / quota / key problems
  // get that provider's notice, never the HF PRO upsell. HF Router models
  // (and direct DeepSeek) keep the HF credits CTA exactly as before.
  const llmCreditCta = !llmHealthError
    ? null
    : usesProviderNotices(llmHealthError.model)
      ? creditCtaForModel(llmHealthError.model, llmHealthError.error, user?.plan)
      : isInferenceCreditError(llmHealthError.error, llmHealthError.errorType)
        ? inferenceCreditCta(user?.plan)
        : null;
  const llmHfCta = llmCreditCta?.kind === 'hf' ? llmCreditCta : null;
  const llmProviderNotice = llmCreditCta && llmCreditCta.kind !== 'hf' ? llmCreditCta : null;

  const trackHealthProClick = () => {
    if (user?.plan === 'pro' || !activeSessionId) return;
    void apiFetch(`/api/pro-click/${activeSessionId}`, {
      method: 'POST',
      body: JSON.stringify({ source: 'llm_health_credit_error', target: 'hf_pro' }),
    }).catch(() => {});
  };

  // -- Welcome screen: no chats here or on the server -----------------------
  // A browser with no local chats waits for the server's lists first, so
  // saved chats open in the normal layout instead of behind WelcomeScreen.
  if (!hasAnySessions) {
    return (
      <Box sx={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
        {initialLoadPending ? <ChatListLoading /> : <WelcomeScreen />}
      </Box>
    );
  }

  // -- Sidebar drawer -----------------------------------------------------
  const sidebarDrawer = (
    <Drawer
      variant={isMobile ? 'temporary' : 'persistent'}
      anchor="left"
      open={isLeftSidebarOpen}
      onClose={() => setLeftSidebarOpen(false)}
      ModalProps={{ keepMounted: true }}
      sx={{
        '& .MuiDrawer-paper': {
          boxSizing: 'border-box',
          width: DRAWER_WIDTH,
          borderRight: '1px solid',
          borderColor: 'divider',
          top: 0,
          height: '100%',
          bgcolor: 'var(--panel)',
        },
      }}
    >
      <SessionSidebar onClose={handleSidebarClose} />
    </Drawer>
  );

  // -- Main chat interface ------------------------------------------------
  return (
    <Box sx={{ display: 'flex', width: '100%', height: '100%' }}>
      {/* -- Left Sidebar ------------------------------------------------- */}
      {isMobile ? (
        sidebarDrawer
      ) : (
        <Box
          component="nav"
          sx={{
            width: isLeftSidebarOpen ? DRAWER_WIDTH : 0,
            flexShrink: 0,
            transition: isResizing.current ? 'none' : 'width 0.2s',
            overflow: 'hidden',
          }}
        >
          {sidebarDrawer}
        </Box>
      )}

      {/* -- Main Content (header + chat + code panel) -------------------- */}
      <Box
        sx={{
          flexGrow: 1,
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          transition: isResizing.current ? 'none' : 'width 0.2s',
          overflow: 'hidden',
          minWidth: 0,
        }}
      >
        {/* -- Top Header Bar --------------------------------------------- */}
        <Box sx={{
          height: { xs: 52, md: 60 },
          px: { xs: 1, md: 2 },
          display: 'flex',
          alignItems: 'center',
          borderBottom: 1,
          borderColor: 'divider',
          bgcolor: 'background.default',
          zIndex: 1200,
          flexShrink: 0,
        }}>
          <IconButton onClick={toggleLeftSidebar} size="small">
            {isLeftSidebarOpen && !isMobile ? <ChevronLeftIcon /> : <MenuIcon />}
          </IconButton>

          <Box sx={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 0.75 }}>
            <Box
              component="img"
              src="/smolagents.webp"
              alt="smolagents"
              sx={{ width: { xs: 20, md: 22 }, height: { xs: 20, md: 22 } }}
            />
            <Typography
              variant="subtitle1"
              sx={{
                fontWeight: 700,
                color: 'var(--text)',
                letterSpacing: '-0.01em',
                fontSize: { xs: '0.88rem', md: '0.95rem' },
              }}
            >
              ML Intern
            </Typography>
          </Box>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <UsageMeter />
            <YoloControl />
            <IconButton
              onClick={toggleTheme}
              size="small"
              sx={{
                color: 'text.secondary',
                '&:hover': { color: 'primary.main' },
              }}
            >
              {themeMode === 'dark' ? <LightModeOutlinedIcon fontSize="small" /> : <DarkModeOutlinedIcon fontSize="small" />}
            </IconButton>

            {user?.picture ? (
              <Avatar
                src={user.picture}
                alt={user.username || 'User'}
                sx={{ width: 28, height: 28, ml: 0.5 }}
              />
            ) : user?.username ? (
              <Avatar
                sx={{
                  width: 28,
                  height: 28,
                  ml: 0.5,
                  bgcolor: 'primary.main',
                  fontSize: '0.75rem',
                  fontWeight: 700,
                }}
              >
                {user.username[0].toUpperCase()}
              </Avatar>
            ) : null}
          </Box>
        </Box>

        {/* -- Chat + Code Panel ------------------------------------------ */}
        <Box
          sx={{
            flexGrow: 1,
            display: 'flex',
            overflow: 'hidden',
          }}
        >
          {/* Chat area */}
          <Box
            component="main"
            className="chat-pane"
            sx={{
              flexGrow: 1,
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
              background: 'var(--body-gradient)',
              p: { xs: 1.5, sm: 2, md: 3 },
              minWidth: 0,
            }}
          >
            {activeSessionId ? (
              // Render ALL sessions — each owns its own useAgentChat.
              // Only the active one renders visible UI (others return null).
              sessions.map((s) => (
                <SessionChat
                  key={s.id}
                  sessionId={s.id}
                  isActive={s.id === activeSessionId}
                  onSessionDead={resolveDeadSession}
                />
              ))
            ) : (
              <Box
                sx={{
                  flex: 1,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexDirection: 'column',
                  gap: 2,
                  px: 2,
                }}
              >
                <Typography variant="h5" color="text.secondary" sx={{ fontFamily: 'monospace', fontSize: { xs: '1rem', md: '1.5rem' } }}>
                  NO SESSION SELECTED
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ fontFamily: 'monospace', fontSize: { xs: '0.75rem', md: '0.875rem' } }}>
                  Initialize a session via the sidebar
                </Typography>
              </Box>
            )}
          </Box>

          {/* Code panel -- inline on desktop, overlay drawer on mobile */}
          {isRightPanelOpen && !isMobile && (
            <>
              <Box
                onMouseDown={startResizing}
                sx={{
                  width: '4px',
                  cursor: 'col-resize',
                  bgcolor: 'divider',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  transition: 'background-color 0.2s',
                  flexShrink: 0,
                  '&:hover': { bgcolor: 'primary.main' },
                }}
              >
                <DragIndicatorIcon
                  sx={{ fontSize: '0.8rem', color: 'text.secondary', pointerEvents: 'none' }}
                />
              </Box>
              <Box
                sx={{
                  width: rightPanelWidth,
                  flexShrink: 0,
                  height: '100%',
                  overflow: 'hidden',
                  borderLeft: '1px solid',
                  borderColor: 'divider',
                  bgcolor: 'var(--panel)',
                }}
              >
                <CodePanel />
              </Box>
            </>
          )}
        </Box>
      </Box>

      {/* Code panel -- drawer overlay on mobile */}
      {isMobile && (
        <Drawer
          anchor="bottom"
          open={isRightPanelOpen}
          onClose={() => useLayoutStore.getState().setRightPanelOpen(false)}
          sx={{
            '& .MuiDrawer-paper': {
              height: '75vh',
              borderTopLeftRadius: 16,
              borderTopRightRadius: 16,
              bgcolor: 'var(--panel)',
            },
          }}
        >
          <CodePanel />
        </Drawer>
      )}
      <Snackbar
        open={showExpiredToast}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        onClose={() => setShowExpiredToast(false)}
      >
        <Alert
          severity="warning"
          variant="filled"
          onClose={() => setShowExpiredToast(false)}
          sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}
        >
          Task expired — create a new task to continue.
        </Alert>
      </Snackbar>
      <Snackbar
        open={!!llmHealthError}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
        onClose={() => setLlmHealthError(null)}
      >
        <Alert
          severity="error"
          variant="filled"
          onClose={() => setLlmHealthError(null)}
          sx={{ fontSize: '0.8rem', maxWidth: 480 }}
        >
          <AlertTitle sx={{ fontWeight: 700, fontSize: '0.85rem' }}>
            {llmProviderNotice ? llmProviderNotice.title : llmErrorTitle}
          </AlertTitle>
          {llmHealthError && (
            <Typography variant="body2" sx={{ fontSize: '0.78rem', opacity: 0.9 }}>
              {llmHealthError.model} — {llmHealthError.error.slice(0, 150)}
            </Typography>
          )}
          {llmProviderNotice && (
            <Typography variant="body2" sx={{ fontSize: '0.78rem', mt: 0.75 }}>
              {llmProviderNotice.message}
            </Typography>
          )}
          {llmHfCta && (
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, mt: 1 }}>
              <Button
                component="a"
                href={llmHfCta.primaryHref}
                target="_blank"
                rel="noopener noreferrer"
                color="inherit"
                size="small"
                variant="outlined"
                onClick={trackHealthProClick}
                sx={{ textTransform: 'none' }}
              >
                {llmHfCta.primaryLabel}
              </Button>
              {llmHfCta.secondaryHref && llmHfCta.secondaryLabel && (
                <Button
                  component="a"
                  href={llmHfCta.secondaryHref}
                  target="_blank"
                  rel="noopener noreferrer"
                  color="inherit"
                  size="small"
                  sx={{ textTransform: 'none' }}
                >
                  {llmHfCta.secondaryLabel}
                </Button>
              )}
            </Box>
          )}
        </Alert>
      </Snackbar>
    </Box>
  );
}

/** A browser with no local chats waits here for the server's chat lists. */
function ChatListLoading() {
  return (
    <Box
      sx={{
        flex: 1,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'var(--body-gradient)',
      }}
    >
      {/* Delayed, so a quick load shows no spinner at all. */}
      <Fade in style={{ transitionDelay: '400ms' }}>
        <CircularProgress size={28} aria-label="Loading chats" sx={{ color: 'var(--muted-text)' }} />
      </Fade>
    </Box>
  );
}
