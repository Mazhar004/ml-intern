import { create } from 'zustand';
import { apiFetch } from '@/utils/api';
import { useSessionStore } from '@/store/sessionStore';
import { isDirectProviderPath } from '@/utils/model';

export interface UsageBucket {
  session_id?: string | null;
  total_usd: number;
  inference_usd: number;
  /** Subset of inference_usd billed to the user's own provider key (Anthropic, DeepSeek). */
  direct_provider_inference_usd: number;
  hf_jobs_estimated_usd: number;
  sandbox_estimated_usd: number;
  llm_calls: number;
  hf_jobs_count: number;
  sandbox_count: number;
  prompt_tokens: number;
  completion_tokens: number;
  cache_read_tokens: number;
  cache_creation_tokens: number;
  total_tokens: number;
  hf_jobs_billable_seconds_estimate: number;
  sandbox_billable_seconds_estimate: number;
}

export interface HfAccountUsageBucket {
  window_start?: string | null;
  window_end?: string | null;
  timezone?: string | null;
  total_usd: number;
  inference_providers_usd: number;
  hf_jobs_usd: number;
  inference_provider_requests: number;
  hf_jobs_minutes: number;
}

export interface HfInferenceProvidersCredits {
  included_usd: number;
  used_usd: number;
  remaining_included_usd: number;
  limit_usd: number;
  remaining_limit_usd: number;
  num_requests: number;
  period_start?: string | null;
  period_end?: string | null;
}

export interface HfAccountUsage {
  source: 'hf_billing';
  available: boolean;
  error?: string | null;
  current_session: HfAccountUsageBucket | null;
  month: HfAccountUsageBucket | null;
  inference_providers_credits: HfInferenceProvidersCredits | null;
}

export interface SessionAutoApprovalUsage {
  enabled: boolean;
  cost_cap_usd?: number | null;
  estimated_spend_usd?: number;
  remaining_usd?: number | null;
}

export interface UsageResponse {
  source: 'app_telemetry';
  currency: 'USD';
  generated_at: string;
  timezone: string;
  session: UsageBucket | null;
  hf_account?: HfAccountUsage | null;
  auto_approval?: SessionAutoApprovalUsage | null;
  links: Record<string, string>;
}

type UsageEventType = 'llm_call' | 'hf_job_complete' | 'sandbox_destroy';

interface UsageStore {
  usage: UsageResponse | null;
  isLoading: boolean;
  error: string | null;
  fetchUsage: (sessionId?: string | null) => Promise<void>;
  applyUsageEvent: (
    sessionId: string,
    eventType: UsageEventType,
    data: Record<string, unknown>,
  ) => void;
}

function numberValue(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function intValue(value: unknown): number {
  return Math.trunc(numberValue(value));
}

function roundUsd(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}

function usageUrl(sessionId?: string | null): string {
  const params = new URLSearchParams();
  const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';
  params.set('tz', timezone);
  if (sessionId) params.set('session_id', sessionId);
  return `/api/usage?${params.toString()}`;
}

function applyEventToBucket(
  bucket: UsageBucket | null,
  eventType: UsageEventType,
  data: Record<string, unknown>,
): UsageBucket | null {
  if (!bucket) return null;
  const next = { ...bucket };

  if (eventType === 'llm_call') {
    const prompt = intValue(data.prompt_tokens);
    const completion = intValue(data.completion_tokens);
    const cacheRead = intValue(data.cache_read_tokens);
    const cacheCreation = intValue(data.cache_creation_tokens);
    const total =
      intValue(data.total_tokens) || prompt + completion + cacheRead + cacheCreation;
    const cost = numberValue(data.cost_usd);
    next.llm_calls += 1;
    next.inference_usd = roundUsd(next.inference_usd + cost);
    // Mirrors aggregate_usage_events in backend/usage.py (direct_provider_for).
    if (typeof data.model === 'string' && isDirectProviderPath(data.model)) {
      next.direct_provider_inference_usd = roundUsd(
        (next.direct_provider_inference_usd ?? 0) + cost,
      );
    }
    next.prompt_tokens += prompt;
    next.completion_tokens += completion;
    next.cache_read_tokens += cacheRead;
    next.cache_creation_tokens += cacheCreation;
    next.total_tokens += total;
  } else if (eventType === 'hf_job_complete') {
    next.hf_jobs_count += 1;
    next.hf_jobs_estimated_usd = roundUsd(
      next.hf_jobs_estimated_usd + numberValue(data.estimated_cost_usd),
    );
    next.hf_jobs_billable_seconds_estimate +=
      intValue(data.billable_seconds_estimate) || intValue(data.wall_time_s);
  }

  next.total_usd = roundUsd(
    next.inference_usd +
      next.hf_jobs_estimated_usd +
      (next.sandbox_estimated_usd ?? 0),
  );
  return next;
}

// Python-style spend coercion from SessionManager._usage_spend_from_response.
function coerceSpend(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? Math.max(0, value) : null;
}

/**
 * Session spend in USD, computed exactly like the backend's
 * SessionManager._usage_spend_from_response (backend/session_manager.py), the
 * figure usage-threshold and YOLO approvals act on. With HF billing, add the
 * telemetry estimates HF never sees (direct-provider calls, jobs, sandboxes);
 * otherwise use the telemetry total, whose inference_usd already includes the
 * direct-provider cost. Undefined only when nothing is loaded yet.
 */
export function sessionSpendUsd(usage: UsageResponse | null | undefined): number | undefined {
  // Partial: the backend reads total_usd only when inference_providers_usd is absent.
  const current: Partial<HfAccountUsageBucket> | null | undefined =
    usage?.hf_account?.current_session;
  const telemetry = usage?.session;
  if (current) {
    const hfSpend = coerceSpend(
      'inference_providers_usd' in current ? current.inference_providers_usd : current.total_usd,
    );
    if (hfSpend != null) {
      return (
        hfSpend +
        (coerceSpend(telemetry?.direct_provider_inference_usd) ?? 0) +
        (coerceSpend(telemetry?.hf_jobs_estimated_usd) ?? 0) +
        (coerceSpend(telemetry?.sandbox_estimated_usd) ?? 0)
      );
    }
  }
  if (!telemetry) return undefined;
  return coerceSpend(telemetry.total_usd) ?? 0;
}

export const useUsageStore = create<UsageStore>()((set, get) => ({
  usage: null,
  isLoading: false,
  error: null,

  fetchUsage: async (sessionId?: string | null) => {
    const current = get().usage;
    set({
      usage:
        sessionId && current?.session?.session_id !== sessionId ? null : current,
      isLoading: true,
      error: null,
    });
    try {
      let response = await apiFetch(usageUrl(sessionId));
      if (response.status === 404 && sessionId) {
        response = await apiFetch(usageUrl());
      }
      if (!response.ok) {
        throw new Error(response.statusText || 'Failed to load usage');
      }
      const usage = (await response.json()) as UsageResponse;
      set({ usage, isLoading: false, error: null });
      if (sessionId && usage.auto_approval) {
        useSessionStore.getState().updateSessionYolo(sessionId, usage.auto_approval);
      }
    } catch (error) {
      set({
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to load usage',
      });
    }
  },

  applyUsageEvent: (sessionId, eventType, data) => {
    if (eventType === 'sandbox_destroy') {
      void get().fetchUsage(sessionId);
      return;
    }

    const current = get().usage;
    if (!current) return;
    set({
      usage: {
        ...current,
        session:
          current.session?.session_id === sessionId
            ? applyEventToBucket(current.session, eventType, data)
            : current.session,
      },
    });
  },
}));
