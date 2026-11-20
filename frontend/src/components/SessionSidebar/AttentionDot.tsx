import { Box } from '@mui/material';

const dotSx = {
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
} as const;

/**
 * Pulsing dot: a chat is waiting for a tool approval. Decorative, so callers
 * also say it in text (e.g. an aria-label) where it matters.
 */
export default function AttentionDot() {
  return <Box aria-hidden sx={dotSx} />;
}
