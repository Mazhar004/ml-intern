import { memo, useEffect, useRef, type MouseEvent, type SyntheticEvent } from 'react';
import { Box, IconButton, Typography } from '@mui/material';
import ChatBubbleOutlineIcon from '@mui/icons-material/ChatBubbleOutline';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import EditOutlinedIcon from '@mui/icons-material/EditOutlined';
import { sessionDisplayTitle } from '@/utils/sessionTitle';
import type { SessionMeta } from '@/types/agent';
import AttentionDot from './AttentionDot';
import SessionTitleEditor from './SessionTitleEditor';

interface SessionRowProps {
  session: SessionMeta;
  /** 1-based position in store order; numbers the "Session NN" fallback title. */
  num: number;
  isSelected: boolean;
  isEditing: boolean;
  onSelect: (sessionId: string) => void;
  onStartRename: (sessionId: string, e: SyntheticEvent) => void;
  onCommitRename: (session: SessionMeta, raw: string, shownTitle: string) => Promise<void>;
  onCancelRename: () => void;
  onDeleteClick: (sessionId: string, e: MouseEvent) => void;
}

const formatTime = (d: string) =>
  new Date(d).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

/** One chat in the sidebar list. Memoized: pass stable callbacks. */
function SessionRow({
  session,
  num,
  isSelected,
  isEditing,
  onSelect,
  onStartRename,
  onCommitRename,
  onCancelRename,
  onDeleteClick,
}: SessionRowProps) {
  const shownTitle = sessionDisplayTitle(session, num);
  const rowRef = useRef<HTMLDivElement>(null);

  // Keep the open chat in view: a new chat can land below the fold (e.g. in
  // the "No project" group, which is listed last) or in a group that opens.
  useEffect(() => {
    if (isSelected) rowRef.current?.scrollIntoView({ block: 'nearest' });
  }, [isSelected]);

  return (
    <Box
      ref={rowRef}
      onClick={() => onSelect(session.id)}
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 1,
        px: 1.5,
        py: 0.875,
        mx: 0.75,
        mb: 0.2,
        // Room around the row when it is scrolled into view (matches the list's padding).
        scrollMarginBlock: '8px',
        borderRadius: '10px',
        cursor: 'pointer',
        transition: 'background-color 0.12s ease',
        bgcolor: isSelected
          ? 'var(--hover-bg)'
          : 'transparent',
        '&:hover': {
          bgcolor: 'var(--hover-bg)',
        },
        '& .delete-btn, & .rename-btn': {
          opacity: 0,
          transition: 'opacity 0.12s',
        },
        '&:hover .delete-btn, &:hover .rename-btn': { opacity: 1 },
        '& .delete-btn:focus-visible, & .rename-btn:focus-visible': { opacity: 1 },
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
        {/* Title row: truncated title, or the inline rename editor */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, minWidth: 0 }}>
          {isEditing ? (
            <SessionTitleEditor
              initialTitle={shownTitle}
              onCommit={(value) => void onCommitRename(session, value, shownTitle)}
              onCancel={onCancelRename}
            />
          ) : (
            <Typography
              variant="body2"
              onDoubleClick={(e) => onStartRename(session.id, e)}
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
              {shownTitle}
            </Typography>
          )}
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
      {session.needsAttention && !isSelected && <AttentionDot />}

      {!isEditing && (
        <IconButton
          className="rename-btn"
          size="small"
          aria-label="Rename chat"
          onClick={(e) => onStartRename(session.id, e)}
          sx={{
            color: 'var(--muted-text)',
            width: 26,
            height: 26,
            flexShrink: 0,
            '&:hover': { color: 'var(--text)', bgcolor: 'var(--hover-bg)' },
          }}
        >
          <EditOutlinedIcon sx={{ fontSize: 15 }} />
        </IconButton>
      )}

      <IconButton
        className="delete-btn"
        size="small"
        aria-label="Delete chat"
        onClick={(e) => onDeleteClick(session.id, e)}
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
}

export default memo(SessionRow);
