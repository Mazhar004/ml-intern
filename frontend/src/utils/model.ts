/**
 * Shared model-id constants used by session-create call sites and the model
 * picker.
 *
 * Keep in sync with DEFAULT_MODEL_OPTIONS in components/Chat/ChatInput.tsx,
 * AVAILABLE_MODELS in agent/core/web_models.py and agent/core/model_ids.py.
 */

export const CLAUDE_OPUS_48_MODEL_PATH = 'anthropic/claude-opus-4.8:fal-ai';
export const GPT_55_MODEL_PATH = 'openai/gpt-5.5:fal-ai';
export const KIMI_K27_CODE_MODEL_PATH = 'moonshotai/Kimi-K2.7-Code:novita';
export const MINIMAX_M3_MODEL_PATH = 'MiniMaxAI/MiniMax-M3:novita';
export const GLM_52_MODEL_PATH = 'zai-org/GLM-5.2:novita';
export const DEEPSEEK_V4_PRO_MODEL_PATH = 'deepseek-ai/DeepSeek-V4-Pro:novita';
// Direct DeepSeek provider ids — route through LiteLLM's ``deepseek/`` provider
// (DEEPSEEK_API_KEY → api.deepseek.com) rather than the HF Router. Mirrors the
// deepseek branch in agent/core/llm_params.py and the fork's config default.
export const DEEPSEEK_V4_PRO_DIRECT_PATH = 'deepseek/deepseek-v4-pro';
export const DEEPSEEK_V4_FLASH_DIRECT_PATH = 'deepseek/deepseek-v4-flash';
// Direct Anthropic API ids (ANTHROPIC_API_KEY → api.anthropic.com). The HF
// Router Claude id above also starts with ``anthropic/``, so match these only
// through isDirectAnthropicPath — never by prefix or substring.
export const CLAUDE_OPUS_55_DIRECT_PATH = 'anthropic/claude-opus-5-5';
export const CLAUDE_FABLE_52_DIRECT_PATH = 'anthropic/claude-fable-5-2';
export const CLAUDE_SONNET_55_DIRECT_PATH = 'anthropic/claude-sonnet-5-5';
export const CLAUDE_HAIKU_55_DIRECT_PATH = 'anthropic/claude-haiku-5-5';
// Direct OpenAI API ids (OPENAI_API_KEY → api.openai.com). The HF Router GPT
// id above also starts with ``openai/``, so match these only through
// isDirectOpenAIPath — never by prefix or substring.
export const GPT_55_DIRECT_PATH = 'openai/gpt-5.5';
export const GPT_62_ASTRA_DIRECT_PATH = 'openai/gpt-6.2-astra';
export const GPT_62_SOL_DIRECT_PATH = 'openai/gpt-6.2-sol';
export const GPT_61_TERRA_DIRECT_PATH = 'openai/gpt-6.1-terra';
export const GPT_6_LUNA_DIRECT_PATH = 'openai/gpt-6-luna';

const DIRECT_ANTHROPIC_PATHS: ReadonlySet<string> = new Set([
  CLAUDE_OPUS_55_DIRECT_PATH,
  CLAUDE_FABLE_52_DIRECT_PATH,
  CLAUDE_SONNET_55_DIRECT_PATH,
  CLAUDE_HAIKU_55_DIRECT_PATH,
]);

const DIRECT_OPENAI_PATHS: ReadonlySet<string> = new Set([
  GPT_55_DIRECT_PATH,
  GPT_62_ASTRA_DIRECT_PATH,
  GPT_62_SOL_DIRECT_PATH,
  GPT_61_TERRA_DIRECT_PATH,
  GPT_6_LUNA_DIRECT_PATH,
]);

/** A provider called with the user's own key (DIRECT_PROVIDER_API_KEY_ENV). */
export type DirectProvider = 'anthropic' | 'openai' | 'deepseek';

/** Who pays for a model: a direct provider's key, or HF credits. */
export type BillingRoute = DirectProvider | 'hf';

const BILLING_LABELS: Readonly<Record<BillingRoute, string>> = {
  hf: 'HF credits',
  anthropic: 'Your Anthropic key',
  openai: 'Your OpenAI key',
  deepseek: 'Your DeepSeek key',
};

export function isClaudePath(modelPath: string | undefined): boolean {
  return !!modelPath && modelPath.includes('anthropic');
}

/**
 * Exact raw-id membership, mirroring is_direct_anthropic_model_id in
 * agent/core/model_ids.py: ``huggingface/...`` and ``:fal-ai`` ids stay on the
 * HF Router.
 */
export function isDirectAnthropicPath(modelPath: string | null | undefined): boolean {
  return !!modelPath && DIRECT_ANTHROPIC_PATHS.has(modelPath);
}

/**
 * Exact raw-id membership, mirroring is_direct_openai_model_id: the router id
 * ``openai/gpt-5.5:fal-ai`` stays on the HF Router.
 */
export function isDirectOpenAIPath(modelPath: string | null | undefined): boolean {
  return !!modelPath && DIRECT_OPENAI_PATHS.has(modelPath);
}

/**
 * The provider whose key pays for ``modelPath``, or null for the HF Router.
 * Mirrors direct_provider_for in agent/core/model_ids.py.
 */
export function directProviderForPath(modelPath: string | null | undefined): DirectProvider | null {
  if (isDirectAnthropicPath(modelPath)) return 'anthropic';
  if (isDirectOpenAIPath(modelPath)) return 'openai';
  if (modelPath?.startsWith('deepseek/')) return 'deepseek';
  return null;
}

export function billingRouteForPath(modelPath: string | null | undefined): BillingRoute {
  return directProviderForPath(modelPath) ?? 'hf';
}

/** Billed to the user's own provider key, not HF credits (direct_provider_for). */
export function isDirectProviderPath(modelPath: string | null | undefined): boolean {
  return directProviderForPath(modelPath) !== null;
}

/** Picker caption saying who pays: "HF credits" or "Your <Provider> key". */
export function billingLabelForPath(modelPath: string | null | undefined): string {
  return BILLING_LABELS[billingRouteForPath(modelPath)];
}
