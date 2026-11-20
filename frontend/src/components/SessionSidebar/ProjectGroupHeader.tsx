import { memo } from 'react';
import { Box } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import FolderOutlinedIcon from '@mui/icons-material/FolderOutlined';
import FolderOffOutlinedIcon from '@mui/icons-material/FolderOffOutlined';
import { describeProjectGroup } from '@/utils/sessionGroups';
import AttentionDot from './AttentionDot';

interface ProjectGroupHeaderProps {
  id: string;
  groupKey: string;
  label: string;
  count: number;
  /** Chats in the group waiting for a tool approval (the open chat excluded). */
  waiting: number;
  /** False for the "No project" group. */
  isProject: boolean;
  expanded: boolean;
  /** id of the element that holds the group's rows. */
  controlsId: string;
  onToggle: (groupKey: string) => void;
}

/** Collapsible sidebar header for one project's chats. */
function ProjectGroupHeader({
  id,
  groupKey,
  label,
  count,
  waiting,
  isProject,
  expanded,
  controlsId,
  onToggle,
}: ProjectGroupHeaderProps) {
  const FolderIcon = isProject ? FolderOutlinedIcon : FolderOffOutlinedIcon;
  // Expanded, each waiting row shows its own dot; collapsed, the header does.
  const shownWaiting = expanded ? 0 : waiting;
  return (
    <Box
      component="button"
      type="button"
      id={id}
      aria-expanded={expanded}
      aria-controls={controlsId}
      aria-label={describeProjectGroup(label, count, shownWaiting)}
      title={label}
      onClick={() => onToggle(groupKey)}
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 0.75,
        width: 'calc(100% - 12px)',
        mx: 0.75,
        px: 1,
        py: 0.625,
        border: 'none',
        borderRadius: '8px',
        bgcolor: 'transparent',
        color: 'var(--muted-text)',
        font: 'inherit',
        fontSize: '0.72rem',
        fontWeight: 600,
        lineHeight: 1.4,
        textAlign: 'left',
        cursor: 'pointer',
        transition: 'background-color 0.12s ease, color 0.12s ease',
        '&:hover': { bgcolor: 'var(--hover-bg)', color: 'var(--text)' },
        '&:focus-visible': {
          outline: '2px solid var(--accent-yellow)',
          outlineOffset: '-2px',
        },
      }}
    >
      <ExpandMoreIcon
        sx={{
          fontSize: 16,
          flexShrink: 0,
          transform: expanded ? 'none' : 'rotate(-90deg)',
          transition: 'transform 0.15s ease',
        }}
      />
      <FolderIcon
        sx={{
          fontSize: 14,
          flexShrink: 0,
          color: isProject ? 'var(--accent-yellow)' : 'inherit',
          opacity: isProject ? 0.85 : 0.6,
        }}
      />
      <Box
        component="span"
        sx={{
          flex: 1,
          minWidth: 0,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
      >
        {label}
      </Box>
      {shownWaiting > 0 && <AttentionDot />}
      <Box
        component="span"
        sx={{
          flexShrink: 0,
          minWidth: 20,
          px: 0.75,
          borderRadius: '999px',
          bgcolor: 'var(--hover-bg)',
          fontSize: '0.65rem',
          fontVariantNumeric: 'tabular-nums',
          textAlign: 'center',
        }}
      >
        {count}
      </Box>
    </Box>
  );
}

export default memo(ProjectGroupHeader);
