/**
 * Shared model-id constants used by session-create call sites and the model
 * picker.
 *
 * Keep in sync with DEFAULT_MODEL_OPTIONS in components/Chat/ChatInput.tsx,
 * AVAILABLE_MODELS in backend/routes/agent.py and agent/core/model_ids.py.
 */

export const CLAUDE_OPUS_48_MODEL_PATH = 'anthropic/claude-opus-4.8:fal-ai';
export const GPT_55_MODEL_PATH = 'openai/gpt-5.5:fal-ai';
export const KIMI_K27_CODE_MODEL_PATH = 'moonshotai/Kimi-K2.7-Code:novita';
export const MINIMAX_M3_MODEL_PATH = 'MiniMaxAI/MiniMax-M3:novita';
export const GLM_52_MODEL_PATH = 'zai-org/GLM-5.2:novita';
export const DEEPSEEK_V4_PRO_MODEL_PATH = 'deepseek-ai/DeepSeek-V4-Pro:novita';
// Direct DeepSeek provider id — routes through LiteLLM's ``deepseek/`` provider
// (DEEPSEEK_API_KEY → api.deepseek.com) rather than the HF Router. Mirrors the
// deepseek branch in agent/core/llm_params.py and the fork's config default.
export const DEEPSEEK_V4_PRO_DIRECT_PATH = 'deepseek/deepseek-v4-pro';
// Direct Anthropic API ids (ANTHROPIC_API_KEY → api.anthropic.com). The HF
// Router Claude id above also starts with ``anthropic/``, so match these only
// through isDirectAnthropicPath — never by prefix or substring.
export const CLAUDE_OPUS_55_DIRECT_PATH = 'anthropic/claude-opus-5-5';
export const CLAUDE_SONNET_55_DIRECT_PATH = 'anthropic/claude-sonnet-5-5';

const DIRECT_ANTHROPIC_PATHS: ReadonlySet<string> = new Set([
  CLAUDE_OPUS_55_DIRECT_PATH,
  CLAUDE_SONNET_55_DIRECT_PATH,
]);

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

/** Billed to the user's own provider key, not HF credits (direct_provider_for). */
export function isDirectProviderPath(modelPath: string | null | undefined): boolean {
  return isDirectAnthropicPath(modelPath) || (!!modelPath && modelPath.startsWith('deepseek/'));
}
