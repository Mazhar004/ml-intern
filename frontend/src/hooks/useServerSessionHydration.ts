/**
 * Loads the server's chat lists (live and saved) into the session store.
 *
 * Runs in AppLayout, which is always mounted. The sidebar only mounts once
 * there is a chat to list, so loading there left a browser with no local
 * chats unable to find its saved ones. Loading retries while the backend is
 * down (see lib/serverSessionLists.ts); if the lists still have not loaded,
 * showing the tab again starts another try. Unmounting cancels it.
 */
import { useEffect, useState } from 'react';
import { createServerListLoader } from '@/lib/serverSessionLists';

export function useServerSessionHydration(): {
  /** True until the first try has an answer: a browser with no local chats
   *  waits for it before choosing between WelcomeScreen and the chat list. */
  initialLoadPending: boolean;
} {
  const [initialLoadPending, setInitialLoadPending] = useState(true);

  useEffect(() => {
    const loader = createServerListLoader({
      onFirstAttemptDone: () => setInitialLoadPending(false),
    });
    void loader.refresh();
    const onVisibilityChange = () => {
      if (document.visibilityState === 'visible') void loader.refresh();
    };
    document.addEventListener('visibilitychange', onVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', onVisibilityChange);
      loader.stop();
    };
  }, []);

  return { initialLoadPending };
}
