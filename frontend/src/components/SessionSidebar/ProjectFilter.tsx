import { memo } from 'react';
import { Box, InputBase, MenuItem, Select, type SelectChangeEvent } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { ALL_PROJECTS, describeProjectGroup } from '@/utils/sessionGroups';
import AttentionDot from './AttentionDot';

export interface ProjectFilterOption {
  /** ALL_PROJECTS or a project group key. */
  value: string;
  label: string;
  count: number;
  /** Chats in the option's group waiting for a tool approval (the open chat excluded). */
  waiting: number;
}

interface ProjectFilterProps {
  value: string;
  options: readonly ProjectFilterOption[];
  onChange: (value: string) => void;
}

const FILTER_LABEL = 'Filter chats by project';

const truncate = {
  flex: 1,
  minWidth: 0,
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
} as const;

// Menu options wrap long folder names instead, so they can be read in full on touch screens.
const wrap = {
  flex: 1,
  minWidth: 0,
  whiteSpace: 'normal',
  overflowWrap: 'anywhere',
} as const;

interface OptionLabelProps {
  label: string;
  count: number;
  /** Show the waiting-for-approval dot. */
  dot: boolean;
  wrapLabel?: boolean;
}

function OptionLabel({ label, count, dot, wrapLabel = false }: OptionLabelProps) {
  return (
    <>
      <Box component="span" sx={wrapLabel ? wrap : truncate}>
        {label}
      </Box>
      {dot && <AttentionDot />}
      <Box
        component="span"
        sx={{
          flexShrink: 0,
          color: 'var(--muted-text)',
          fontSize: '0.7rem',
          fontVariantNumeric: 'tabular-nums',
        }}
      >
        {count}
      </Box>
    </>
  );
}

/** Sidebar dropdown that narrows the chat list to one project. */
function ProjectFilter({ value, options, onChange }: ProjectFilterProps) {
  // Chats waiting for approval in the projects this filter hides.
  const waitingElsewhere =
    value === ALL_PROJECTS
      ? 0
      : options.reduce((sum, option) => (option.value === value ? sum : sum + option.waiting), 0);

  const renderValue = (selected: string) => {
    const option = options.find((o) => o.value === selected);
    return option ? (
      <OptionLabel label={option.label} count={option.count} dot={waitingElsewhere > 0} />
    ) : null;
  };

  return (
    <Select<string>
      value={value}
      onChange={(event: SelectChangeEvent<string>) => onChange(event.target.value)}
      renderValue={renderValue}
      input={<InputBase />}
      IconComponent={ExpandMoreIcon}
      inputProps={{
        'aria-label':
          waitingElsewhere > 0
            ? `${FILTER_LABEL}, ${waitingElsewhere} waiting for approval in other projects`
            : FILTER_LABEL,
      }}
      MenuProps={{
        MenuListProps: { 'aria-label': 'Projects' },
        slotProps: {
          paper: {
            sx: {
              mt: 0.5,
              maxHeight: 'min(360px, calc(100% - 32px))',
              maxWidth: 'min(320px, calc(100% - 32px))',
              bgcolor: 'var(--panel)',
              border: '1px solid var(--border-hover)',
              borderRadius: '10px',
              boxShadow: 'var(--shadow-1)',
            },
          },
        },
      }}
      sx={{
        width: '100%',
        fontSize: '0.78rem',
        color: 'var(--text)',
        bgcolor: 'var(--hover-bg)',
        border: '1px solid var(--border)',
        borderRadius: '8px',
        transition: 'border-color 0.12s ease',
        '&:hover': { borderColor: 'var(--border-hover)' },
        '&.Mui-focused': { borderColor: 'var(--accent-yellow)', boxShadow: 'var(--focus)' },
        '& .MuiSelect-select': {
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          py: 0.75,
          pl: 1.25,
          minHeight: 0,
        },
        '& .MuiSelect-icon': { color: 'var(--muted-text)', fontSize: 18 },
      }}
    >
      {options.map((option) => (
        <MenuItem
          key={option.value}
          value={option.value}
          dense
          aria-label={describeProjectGroup(option.label, option.count, option.waiting)}
          sx={{
            gap: 1.5,
            fontSize: '0.8rem',
            color: 'var(--text)',
            '&.Mui-selected': { bgcolor: 'var(--accent-yellow-weak)' },
            '&:hover, &.Mui-selected:hover': { bgcolor: 'var(--hover-bg)' },
            // Keyboard focus gets its own ring, so it never looks like the selected tint.
            '&.Mui-focusVisible': {
              bgcolor: 'var(--hover-bg)',
              outline: '2px solid var(--accent-yellow)',
              outlineOffset: '-2px',
            },
            '&.Mui-selected.Mui-focusVisible': { bgcolor: 'var(--accent-yellow-weak)' },
          }}
        >
          <OptionLabel
            label={option.label}
            count={option.count}
            dot={option.waiting > 0}
            wrapLabel
          />
        </MenuItem>
      ))}
    </Select>
  );
}

export default memo(ProjectFilter);
