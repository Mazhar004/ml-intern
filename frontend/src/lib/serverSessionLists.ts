/**
 * Loads the server's chat lists into the session store, retrying while the
 * backend cannot answer.
 *
 * A run fetches GET /api/sessions (chats live in the backend) and GET
 * /api/sessions/archived (chats saved on disk), then merges them in that
 * order, so the archived pass de-dupes against the live rows. A failed
 * request (network error, timeout or non-2xx) is retried after 1s, 2s, 4s,
 * 8s, then every 15s, for about two minutes. A 401/403 ends the run: the
 * visitor is signed out, and WelcomeScreen handles sign-in. Both merges are
 * idempotent, so merging a list twice adds no rows.
 */
import { useSessionStore } from '@/store/sessionStore';
import { logger } from '@/utils/logger';

const LIVE_SESSIONS_PATH = '/api/sessions';
const ARCHIVED_SESSIONS_PATH = '/api/sessions/archived';

/** Waits before the first retries, in order; every later retry waits STEADY_RETRY_MS. */
const BACKOFF_MS: readonly number[] = [1_000, 2_000, 4_000, 8_000];
const STEADY_RETRY_MS = 15_000;
/** A run stops retrying once its waits would add up to more than this. */
const RETRY_BUDGET_MS = 120_000;
/** A request with no answer by then counts as failed. Generous, because the
 *  archived listing reads every saved chat file. */
const REQUEST_TIMEOUT_MS = 30_000;

type StoreState = ReturnType<typeof useSessionStore.getState>;
type LiveSessionRow = Parameters<StoreState['mergeServerSessions']>[0][number];
type ArchivedSessionRow = Parameters<StoreState['mergeArchivedSessions']>[0][number];
type FetchFn = (path: string, init: RequestInit) => Promise<Response>;
type WaitFn = (ms: number, signal: AbortSignal) => Promise<void>;

/** How a run ended. 'failed': it gave up after retrying for about two minutes. */
export type ServerListsOutcome = 'merged' | 'failed' | 'signed-out';

interface SessionIdRow {
  session_id: string;
}

type ListResult =
  | { status: 'ok'; rows: SessionIdRow[] }
  | { status: 'skipped' }
  | { status: 'signed-out' }
  | { status: 'failed'; reason: unknown };
type FailedList = Extract<ListResult, { status: 'failed' }>;

const SKIPPED: Promise<ListResult> = Promise.resolve({ status: 'skipped' });

interface Progress {
  liveMerged: boolean;
  archivedMerged: boolean;
}

interface AttemptResult {
  progress: Progress;
  outcome: 'merged' | 'signed-out' | 'retry';
  /** Why the attempt fell short, for the log. */
  reason?: unknown;
}

export interface ServerListLoaderOptions {
  /** Called once, when the first attempt has an answer (whatever it is). */
  onFirstAttemptDone?: () => void;
  /** Test seams: the request function, and the wait between attempts. */
  fetch?: FetchFn;
  wait?: WaitFn;
}

export interface ServerListLoader {
  /** Load the lists unless they already merged; joins a run in progress.
   *  Resolves with how the run ended, or null if it was stopped. */
  refresh: () => Promise<ServerListsOutcome | null>;
  /** Stop the run in progress: abort its requests and drop its retries. */
  stop: () => void;
}

/** Wait before retry number `retry` (0-based). */
function retryDelayMs(retry: number): number {
  return BACKOFF_MS[retry] ?? STEADY_RETRY_MS;
}

/** Resolves after `ms`, or as soon as `signal` aborts. */
function delay(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const finish = () => {
      clearTimeout(timer);
      signal.removeEventListener('abort', finish);
      resolve();
    };
    const timer = setTimeout(finish, ms);
    signal.addEventListener('abort', finish);
  });
}

function isSessionIdRow(row: unknown): row is SessionIdRow {
  if (typeof row !== 'object' || row === null) return false;
  const id = (row as { session_id?: unknown }).session_id;
  return typeof id === 'string' && id !== '';
}

/** GET one list. Never throws: a failure comes back as a result. */
async function fetchList(path: string, signal: AbortSignal, fetchFn: FetchFn): Promise<ListResult> {
  const request = new AbortController();
  const abort = () => request.abort();
  const timeout = setTimeout(abort, REQUEST_TIMEOUT_MS);
  signal.addEventListener('abort', abort);
  try {
    // Plain fetch rather than apiFetch: a 401 here must not send a signed-out
    // visitor to the login page. WelcomeScreen handles sign-in.
    const response = await fetchFn(path, { credentials: 'include', signal: request.signal });
    if (response.status === 401 || response.status === 403) return { status: 'signed-out' };
    if (!response.ok) {
      return { status: 'failed', reason: new Error(`GET ${path} returned ${response.status}`) };
    }
    const body: unknown = await response.json();
    if (!Array.isArray(body)) {
      return { status: 'failed', reason: new Error(`GET ${path} did not return a list`) };
    }
    return { status: 'ok', rows: body.filter(isSessionIdRow) };
  } catch (error) {
    return { status: 'failed', reason: error };
  } finally {
    clearTimeout(timeout);
    signal.removeEventListener('abort', abort);
  }
}

/**
 * One try at both lists: fetch what is still missing (both at once) and merge
 * each as soon as it may go in. The archived pass de-dupes against live rows,
 * so it waits for the live list, and the archived list is fetched again
 * whenever the live one still has to be merged. Null when the run was stopped.
 */
async function attemptLoad(
  progress: Progress,
  signal: AbortSignal,
  fetchFn: FetchFn,
): Promise<AttemptResult | null> {
  const needLive = !progress.liveMerged;
  const needArchived = needLive || !progress.archivedMerged;
  // fetchList never rejects, so awaiting these one after the other is safe.
  const livePending = needLive ? fetchList(LIVE_SESSIONS_PATH, signal, fetchFn) : SKIPPED;
  const archivedPending = needArchived ? fetchList(ARCHIVED_SESSIONS_PATH, signal, fetchFn) : SKIPPED;

  // Rows go in as the server sent them; the merges read each field defensively.
  const live = await livePending;
  if (signal.aborted) return null;
  if (live.status === 'ok') {
    useSessionStore.getState().mergeServerSessions(live.rows as LiveSessionRow[]);
  }
  const archived = await archivedPending;
  if (signal.aborted) return null;
  if (archived.status === 'ok') {
    const store = useSessionStore.getState();
    store.mergeArchivedSessions(archived.rows as ArchivedSessionRow[]);
    store.setArchivedSessionIds(archived.rows.map((row) => row.session_id));
  }

  const next: Progress = {
    liveMerged: progress.liveMerged || live.status === 'ok',
    archivedMerged: archived.status === 'ok' || (progress.archivedMerged && !needArchived),
  };
  if (next.liveMerged && next.archivedMerged) return { progress: next, outcome: 'merged' };
  const results = [live, archived];
  if (results.some((r) => r.status === 'signed-out')) {
    return { progress: next, outcome: 'signed-out' };
  }
  const failure = results.find((r): r is FailedList => r.status === 'failed');
  return { progress: next, outcome: 'retry', reason: failure?.reason };
}

/** Try until both lists are merged, the visitor is signed out, or the retry
 *  budget is spent. Null when stopped. */
async function runLoad(
  signal: AbortSignal,
  fetchFn: FetchFn,
  wait: WaitFn,
  onAttemptDone: () => void,
): Promise<ServerListsOutcome | null> {
  const startedAt = Date.now();
  let progress: Progress = { liveMerged: false, archivedMerged: false };
  let waitedMs = 0;
  for (let retry = 0; ; retry += 1) {
    const result = await attemptLoad(progress, signal, fetchFn);
    if (!result) return null;
    onAttemptDone();
    if (result.outcome !== 'retry') return result.outcome;
    progress = result.progress;
    const delayMs = retryDelayMs(retry);
    // Count slow requests and throttled timers too, not just the waits.
    const spentMs = Math.max(waitedMs, Date.now() - startedAt);
    if (spentMs + delayMs > RETRY_BUDGET_MS) {
      logger.warn('Could not load the chat list from the server:', result.reason);
      return 'failed';
    }
    logger.log(`Chat list not loaded; retrying in ${delayMs / 1000}s:`, result.reason);
    await wait(delayMs, signal);
    if (signal.aborted) return null;
    waitedMs += delayMs;
  }
}

interface RunningLoad {
  controller: AbortController;
  done: Promise<ServerListsOutcome | null>;
}

/** A loader for one page load. Once the lists have merged, refresh() is a no-op. */
export function createServerListLoader(options: ServerListLoaderOptions = {}): ServerListLoader {
  const fetchFn: FetchFn = options.fetch ?? ((path, init) => fetch(path, init));
  const wait = options.wait ?? delay;
  let running: RunningLoad | null = null;
  let merged = false;
  let firstAttemptDone = false;

  const markFirstAttemptDone = () => {
    if (firstAttemptDone) return;
    firstAttemptDone = true;
    options.onFirstAttemptDone?.();
  };

  const start = (): Promise<ServerListsOutcome | null> => {
    const controller = new AbortController();
    const done = runLoad(controller.signal, fetchFn, wait, markFirstAttemptDone)
      .catch((error: unknown): ServerListsOutcome => {
        logger.error('Loading the chat list failed:', error);
        return 'failed';
      })
      .then((outcome) => {
        if (running?.controller === controller) running = null;
        if (controller.signal.aborted) return null;
        markFirstAttemptDone();
        if (outcome === 'merged') {
          merged = true;
          useSessionStore.getState().setServerListsMerged(true);
        }
        return outcome;
      });
    running = { controller, done };
    return done;
  };

  return {
    refresh: () => {
      if (merged) return Promise.resolve('merged');
      return running ? running.done : start();
    },
    stop: () => {
      running?.controller.abort();
      running = null;
    },
  };
}
