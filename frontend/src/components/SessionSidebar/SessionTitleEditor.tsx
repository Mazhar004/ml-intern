import { useRef, useState } from 'react';
import { TextField } from '@mui/material';
import { MAX_SESSION_TITLE_CHARS } from '@/utils/sessionTitle';

interface SessionTitleEditorProps {
  initialTitle: string;
  /** Called once with the raw input on Enter or blur. */
  onCommit: (value: string) => void;
  /** Called once on Escape. */
  onCancel: () => void;
}

const stop = (e: React.SyntheticEvent) => e.stopPropagation();

/** Inline sidebar title editor: Enter/blur saves, Escape cancels. Clicks
 *  inside never reach the row, so editing doesn't switch sessions. */
export default function SessionTitleEditor({
  initialTitle,
  onCommit,
  onCancel,
}: SessionTitleEditorProps) {
  const [value, setValue] = useState(initialTitle);
  // Enter/Escape unmount the editor, which also fires blur — finish once.
  const finished = useRef(false);

  const finish = (action: () => void) => {
    if (finished.current) return;
    finished.current = true;
    action();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    e.stopPropagation();
    if (e.key === 'Enter') {
      e.preventDefault();
      finish(() => onCommit(value));
    } else if (e.key === 'Escape') {
      e.preventDefault();
      finish(onCancel);
    }
  };

  return (
    <TextField
      autoFocus
      variant="standard"
      size="small"
      value={value}
      onChange={(e) => setValue(e.target.value)}
      onFocus={(e) => e.target.select()}
      onKeyDown={handleKeyDown}
      onBlur={() => finish(() => onCommit(value))}
      onClick={stop}
      onMouseDown={stop}
      onDoubleClick={stop}
      inputProps={{ maxLength: MAX_SESSION_TITLE_CHARS, 'aria-label': 'Chat name' }}
      sx={{
        flex: 1,
        minWidth: 0,
        '& .MuiInputBase-root': {
          fontSize: '0.84rem',
          lineHeight: 1.4,
          color: 'var(--text)',
        },
        '& input': { py: 0, height: 'auto' },
      }}
    />
  );
}
