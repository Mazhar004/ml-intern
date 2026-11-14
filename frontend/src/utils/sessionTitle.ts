/**
 * Chat title helpers shared by the sidebar rename flow.
 *
 * Mirrors the backend rules in backend/models.py (normalize_session_title):
 * control characters dropped, whitespace runs collapsed, max 80 chars.
 */
import type { SessionMeta } from '@/types/agent';
import { apiFetch } from '@/utils/api';

export const MAX_SESSION_TITLE_CHARS = 80;
const RENAME_FAILED = 'Could not rename chat.';

// eslint-disable-next-line no-control-regex
const CONTROL_CHARS = /[\u0000-\u0008\u000E-\u001F\u007F-\u009F]/g;

export function normalizeSessionTitle(raw: string): string {
  return raw.replace(CONTROL_CHARS, '').replace(/\s+/g, ' ').trim();
}

/** What the sidebar shows: untitled "Chat N" rows read as "Session NN". */
export function sessionDisplayTitle(session: SessionMeta, num: number): string {
  if (!session.titleIsCustom && session.title.startsWith('Chat ')) {
    return `Session ${String(num).padStart(2, '0')}`;
  }
  return session.title;
}

async function errorDetail(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    const detail = (body as { detail?: unknown })?.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail) && typeof detail[0]?.msg === 'string') {
      return detail[0].msg.replace(/^Value error, /, '');
    }
  } catch {
    /* non-JSON error body */
  }
  return RENAME_FAILED;
}

/**
 * PATCH the title and return the stored value. Tries the row's own id
 * first (live session, or the archive id for archived rows), then the
 * original archive id when a "Catch me up" row's live id is gone.
 */
export async function saveSessionTitle(session: SessionMeta, title: string): Promise<string> {
  const ids = Array.from(
    new Set([session.id, session.sourceArchiveId].filter((id): id is string => Boolean(id))),
  );
  let message = RENAME_FAILED;
  for (const id of ids) {
    const response = await apiFetch(`/api/session/${encodeURIComponent(id)}/title`, {
      method: 'PATCH',
      body: JSON.stringify({ title }),
    });
    if (response.ok) {
      const data: { title?: unknown } = await response.json();
      return typeof data.title === 'string' ? data.title : title;
    }
    message = await errorDetail(response);
    if (response.status !== 404) break;
  }
  throw new Error(message);
}
