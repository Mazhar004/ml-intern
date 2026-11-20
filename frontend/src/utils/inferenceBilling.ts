import { directProviderForPath, type DirectProvider } from '@/utils/model';

export const INFERENCE_PROVIDERS_PRICING_URL = 'https://huggingface.co/docs/inference-providers/pricing';
export const HF_PRO_SUBSCRIBE_URL = 'https://huggingface.co/subscribe/pro';

export type PlanTier = 'free' | 'pro';

/** HF Inference Providers upsell (PRO / pay-as-you-go links). */
export interface HfCreditCta {
  kind: 'hf';
  title: string;
  message: string;
  primaryLabel: string;
  primaryHref: string;
  secondaryLabel: string | null;
  secondaryHref: string | null;
}

/** Message-only notice for a model billed to the user's own provider key. */
export interface ProviderNotice {
  kind: 'anthropic' | 'openai';
  title: string;
  message: string;
}

export type CreditCta = HfCreditCta | ProviderNotice;

// Most errors arrive as the agent loop's friendly text: keep these in sync with
// _ANTHROPIC_BILLING_MESSAGE, _ANTHROPIC_AUTH_MESSAGE, _OPENAI_QUOTA_MESSAGE and
// _OPENAI_AUTH_MESSAGE in agent/core/agent_loop.py, and with
// _MISSING_ANTHROPIC_KEY_MESSAGE and _MISSING_OPENAI_KEY_MESSAGE in
// agent/core/llm_params.py. A failed pre-call compaction summary and the LLM
// health probe bypass that mapping and surface LiteLLM's raw text, so each
// provider's own wording is matched too. Errors are lowercased before
// matching. Only direct Anthropic and OpenAI sessions reach these checks.
const ANTHROPIC_BILLING_PHRASES = [
  'your anthropic api credit balance is too low',
  'credit balance is too low to access the anthropic api',
];
const ANTHROPIC_KEY_PHRASES = [
  'anthropic rejected the api key',
  'claude via the anthropic api needs anthropic_api_key',
  'invalid x-api-key',
  'authentication_error',
];
const OPENAI_QUOTA_PHRASES = [
  'your openai api quota is exhausted',
  'insufficient_quota',
  'exceeded your current quota',
];
const OPENAI_KEY_PHRASES = [
  'openai rejected the api key',
  'gpt via the openai api needs openai_api_key',
  'incorrect api key provided',
  'invalid_api_key',
];

const ANTHROPIC_CREDIT_NOTICE: ProviderNotice = {
  kind: 'anthropic',
  title: 'Anthropic credit balance too low',
  message: 'Your Anthropic API credit balance is too low. Add credits in the Anthropic Console (console.anthropic.com) under Billing.',
};

const ANTHROPIC_KEY_NOTICE: ProviderNotice = {
  kind: 'anthropic',
  title: 'Anthropic API key missing or invalid',
  message: 'Set a valid ANTHROPIC_API_KEY in the repo .env and restart the backend.',
};

const OPENAI_QUOTA_NOTICE: ProviderNotice = {
  kind: 'openai',
  title: 'OpenAI quota exhausted',
  message: 'Your OpenAI API quota is exhausted. Add credits or raise your usage limits in the OpenAI dashboard (platform.openai.com) under Billing.',
};

const OPENAI_KEY_NOTICE: ProviderNotice = {
  kind: 'openai',
  title: 'OpenAI API key missing or invalid',
  message: 'Set a valid OPENAI_API_KEY in the repo .env and restart the backend.',
};

// The backend's own direct-provider wording (agent_loop / llm_params messages
// above). Only direct sessions produce it, but a chat error banner re-renders
// with the session's current model: after a switch to an HF model these keep
// their provider notice instead of turning into the HF upsell.
const BACKEND_PROVIDER_MESSAGE_NOTICES: ReadonlyArray<readonly [string, ProviderNotice]> = [
  ['your anthropic api credit balance is too low', ANTHROPIC_CREDIT_NOTICE],
  ['anthropic rejected the api key', ANTHROPIC_KEY_NOTICE],
  ['claude via the anthropic api needs anthropic_api_key', ANTHROPIC_KEY_NOTICE],
  ['your openai api quota is exhausted', OPENAI_QUOTA_NOTICE],
  ['openai rejected the api key', OPENAI_KEY_NOTICE],
  ['gpt via the openai api needs openai_api_key', OPENAI_KEY_NOTICE],
];

function backendProviderNotice(error: string | undefined): ProviderNotice | null {
  const value = (error ?? '').toLowerCase();
  const match = BACKEND_PROVIDER_MESSAGE_NOTICES.find(([phrase]) => value.includes(phrase));
  return match ? match[1] : null;
}

export function isInferenceCreditError(error: string | undefined, errorType?: string): boolean {
  if (errorType === 'credits') return true;
  const value = (error ?? '').toLowerCase();
  return (
    value.includes('402') ||
    (value.includes('credit') && (
      value.includes('insufficient') ||
      value.includes('exhausted') ||
      value.includes('out of') ||
      value.includes('billing')
    ))
  );
}

export function inferenceCreditCta(plan: PlanTier | undefined): HfCreditCta {
  if (plan === 'pro') {
    return {
      kind: 'hf',
      title: 'Inference credits exhausted',
      message: 'Your HF account needs more Inference Providers credits before this model can continue.',
      primaryLabel: 'View pricing',
      primaryHref: INFERENCE_PROVIDERS_PRICING_URL,
      secondaryLabel: null,
      secondaryHref: null,
    };
  }

  return {
    kind: 'hf',
    title: 'Inference credits exhausted',
    message: 'Upgrade to HF PRO for more monthly Inference Providers usage, or review pay-as-you-go pricing.',
    primaryLabel: 'Upgrade to PRO',
    primaryHref: HF_PRO_SUBSCRIBE_URL,
    secondaryLabel: 'View pricing',
    secondaryHref: INFERENCE_PROVIDERS_PRICING_URL,
  };
}

const mentionsAny = (value: string, phrases: readonly string[]) => (
  phrases.some((phrase) => value.includes(phrase))
);

function anthropicNotice(error: string | undefined): ProviderNotice | null {
  const value = (error ?? '').toLowerCase();
  if (mentionsAny(value, ANTHROPIC_BILLING_PHRASES)) return ANTHROPIC_CREDIT_NOTICE;
  if (mentionsAny(value, ANTHROPIC_KEY_PHRASES)) return ANTHROPIC_KEY_NOTICE;
  return null;
}

function openaiNotice(error: string | undefined): ProviderNotice | null {
  const value = (error ?? '').toLowerCase();
  if (mentionsAny(value, OPENAI_QUOTA_PHRASES)) return OPENAI_QUOTA_NOTICE;
  if (mentionsAny(value, OPENAI_KEY_PHRASES)) return OPENAI_KEY_NOTICE;
  return null;
}

type ProviderNoticeFor = (error: string | undefined) => ProviderNotice | null;

// Providers whose failures get their own notice instead of the HF upsell.
// Direct DeepSeek is left out on purpose: it keeps the HF credits CTA gating.
const PROVIDER_NOTICES: Partial<Record<DirectProvider, ProviderNoticeFor>> = {
  anthropic: anthropicNotice,
  openai: openaiNotice,
};

function providerNoticeFor(modelPath: string | null | undefined): ProviderNoticeFor | undefined {
  const provider = directProviderForPath(modelPath);
  return provider ? PROVIDER_NOTICES[provider] : undefined;
}

/**
 * True when a failed call on ``modelPath`` (direct Anthropic or OpenAI) gets a
 * provider notice from creditCtaForModel, never the HF credits upsell.
 */
export function usesProviderNotices(modelPath: string | null | undefined): boolean {
  return providerNoticeFor(modelPath) !== undefined;
}

/**
 * Banner CTA for a failed call on ``modelPath``. Direct Anthropic and OpenAI
 * models get that provider's notice, or nothing (never the HF upsell). Any
 * other model shows a provider notice for the backend's own direct-provider
 * messages, and otherwise keeps the HF credits CTA gating unchanged (direct
 * DeepSeek included).
 */
export function creditCtaForModel(
  modelPath: string | null | undefined,
  error: string | undefined,
  plan: PlanTier | undefined,
): CreditCta | null {
  const providerNotice = providerNoticeFor(modelPath);
  if (providerNotice) return providerNotice(error);
  const backendNotice = backendProviderNotice(error);
  if (backendNotice) return backendNotice;
  return isInferenceCreditError(error) ? inferenceCreditCta(plan) : null;
}
