import { isDirectAnthropicPath } from '@/utils/model';

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

/** Message-only notice for a model billed to the user's own Anthropic key. */
export interface AnthropicNotice {
  kind: 'anthropic';
  title: string;
  message: string;
}

export type CreditCta = HfCreditCta | AnthropicNotice;

// Most errors arrive as the agent loop's friendly text: keep these in sync with
// _ANTHROPIC_BILLING_MESSAGE and _ANTHROPIC_AUTH_MESSAGE in
// agent/core/agent_loop.py and _MISSING_ANTHROPIC_KEY_MESSAGE in
// agent/core/llm_params.py. A failed pre-call compaction summary bypasses that
// mapping and surfaces LiteLLM's raw text, so Anthropic's own wording is
// matched too. Only direct Anthropic sessions reach these checks.
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

const ANTHROPIC_CREDIT_NOTICE: AnthropicNotice = {
  kind: 'anthropic',
  title: 'Anthropic credit balance too low',
  message: 'Your Anthropic API credit balance is too low. Add credits in the Anthropic Console (console.anthropic.com) under Billing.',
};

const ANTHROPIC_KEY_NOTICE: AnthropicNotice = {
  kind: 'anthropic',
  title: 'Anthropic API key missing or invalid',
  message: 'Set a valid ANTHROPIC_API_KEY in the repo .env and restart the backend.',
};

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

function anthropicNotice(error: string | undefined): AnthropicNotice | null {
  const value = (error ?? '').toLowerCase();
  if (ANTHROPIC_BILLING_PHRASES.some((phrase) => value.includes(phrase))) return ANTHROPIC_CREDIT_NOTICE;
  if (ANTHROPIC_KEY_PHRASES.some((phrase) => value.includes(phrase))) return ANTHROPIC_KEY_NOTICE;
  return null;
}

/**
 * Banner CTA for a failed call on ``modelPath``. Direct Anthropic models get an
 * Anthropic notice (never the HF upsell); every other model, including direct
 * DeepSeek, keeps the HF credits CTA gating unchanged.
 */
export function creditCtaForModel(
  modelPath: string | null | undefined,
  error: string | undefined,
  plan: PlanTier | undefined,
): CreditCta | null {
  if (isDirectAnthropicPath(modelPath)) return anthropicNotice(error);
  return isInferenceCreditError(error) ? inferenceCreditCta(plan) : null;
}
