import { useId, type ReactNode } from 'react';
import { Box } from '@mui/material';
import ProjectGroupHeader from './ProjectGroupHeader';

interface ProjectGroupSectionProps {
  groupKey: string;
  label: string;
  count: number;
  /** Chats in the group waiting for a tool approval (the open chat excluded). */
  waiting: number;
  /** False for the "No project" group. */
  isProject: boolean;
  expanded: boolean;
  onToggle: (groupKey: string) => void;
  /** The group's rows. Pass them only while expanded. */
  children?: ReactNode;
}

/** A project's header plus the region holding its rows (hidden when collapsed). */
export default function ProjectGroupSection({
  groupKey,
  label,
  count,
  waiting,
  isProject,
  expanded,
  onToggle,
  children,
}: ProjectGroupSectionProps) {
  const baseId = useId();
  const headerId = `${baseId}-header`;
  const rowsId = `${baseId}-rows`;

  return (
    <Box sx={{ mb: 0.5 }}>
      <ProjectGroupHeader
        id={headerId}
        groupKey={groupKey}
        label={label}
        count={count}
        waiting={waiting}
        isProject={isProject}
        expanded={expanded}
        controlsId={rowsId}
        onToggle={onToggle}
      />
      <Box id={rowsId} role="group" aria-labelledby={headerId} hidden={!expanded}>
        {children}
      </Box>
    </Box>
  );
}
