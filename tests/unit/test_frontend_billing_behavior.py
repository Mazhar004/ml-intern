"""Run the real frontend routing and billing helpers against the backend.

``test_frontend_model_paths`` reads the TS declarations; this file executes
``utils/model.ts`` and ``utils/inferenceBilling.ts``. They are bundled with
the esbuild that ships with Vite (no new dependency) and called from node, so
routing, captions and banner notices are checked against
``direct_provider_for`` and the backend's own error messages. Skipped when
node or the frontend dependencies are not installed.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from agent.core.agent_loop import (
    _ANTHROPIC_AUTH_MESSAGE,
    _ANTHROPIC_BILLING_MESSAGE,
    _OPENAI_AUTH_MESSAGE,
    _OPENAI_QUOTA_MESSAGE,
)
from agent.core.llm_params import (
    _MISSING_ANTHROPIC_KEY_MESSAGE,
    _MISSING_OPENAI_KEY_MESSAGE,
)
from agent.core.model_ids import direct_provider_for
from agent.core.web_models import AVAILABLE_MODELS

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(
    NODE is None or not (FRONTEND / "node_modules" / "esbuild").is_dir(),
    reason="needs node and frontend/node_modules (npm ci)",
)

_RUNNER = """
import { createRequire } from 'node:module';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { readFileSync } from 'node:fs';

const [frontendDir, workDir] = process.argv.slice(2);
const require = createRequire(path.join(frontendDir, 'package.json'));
const esbuild = require('esbuild');
const src = path.join(frontendDir, 'src');
const outfile = path.join(workDir, 'bundle.mjs');
await esbuild.build({
  stdin: {
    contents: [
      `export * as M from ${JSON.stringify(path.join(src, 'utils/model.ts'))};`,
      `export * as B from ${JSON.stringify(path.join(src, 'utils/inferenceBilling.ts'))};`,
    ].join('\\n'),
    resolveDir: workDir,
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  platform: 'node',
  outfile,
  alias: { '@': src },
  logLevel: 'error',
});
const mod = await import(pathToFileURL(outfile).href);
const calls = JSON.parse(readFileSync(0, 'utf8'));
const results = calls.map(([name, args]) => {
  const [ns, fn] = name.split('.');
  const value = mod[ns][fn](...args);
  return value === undefined ? null : value;
});
process.stdout.write(JSON.stringify(results));
"""

HF_MODELS = (
    "zai-org/GLM-5.2:novita",
    "openai/gpt-5.5:fal-ai",
    "anthropic/claude-opus-4.8:fal-ai",
)
COLLISION_IDS = (
    "huggingface/openai/gpt-5.5",
    "huggingface/anthropic/claude-opus-5-5",
    "openai/gpt-5.4",
    "openai/responses/gpt-6.2-sol",
    "anthropic/claude-opus-4-7",
    "deepseek-ai/DeepSeek-V4-Pro:novita",
    "",
)
CAPTIONS = {
    None: "HF credits",
    "anthropic": "Your Anthropic key",
    "openai": "Your OpenAI key",
    "deepseek": "Your DeepSeek key",
}


@pytest.fixture(scope="module")
def run_ts(tmp_path_factory):
    work_dir = tmp_path_factory.mktemp("frontend_bundle")
    runner = work_dir / "runner.mjs"
    runner.write_text(_RUNNER)

    def call(calls: list[tuple[str, list]]) -> list:
        completed = subprocess.run(
            [NODE, str(runner), str(FRONTEND), str(work_dir)],
            input=json.dumps(calls),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        return json.loads(completed.stdout)

    return call


def _picker_ids() -> list[str]:
    return [model["id"] for model in AVAILABLE_MODELS]


def test_routing_and_captions_match_the_backend(run_ts):
    ids = _picker_ids() + list(COLLISION_IDS)
    calls = [("M.directProviderForPath", [model_id]) for model_id in ids]
    calls += [("M.billingLabelForPath", [model_id]) for model_id in ids]

    results = run_ts(calls)

    routes, captions = results[: len(ids)], results[len(ids) :]
    assert routes == [direct_provider_for(model_id) for model_id in ids]
    assert captions == [CAPTIONS[direct_provider_for(model_id)] for model_id in ids]


_PROVIDER_MESSAGES = {
    "anthropic": (
        _ANTHROPIC_BILLING_MESSAGE,
        _ANTHROPIC_AUTH_MESSAGE,
        _MISSING_ANTHROPIC_KEY_MESSAGE,
    ),
    "openai": (
        _OPENAI_QUOTA_MESSAGE,
        _OPENAI_AUTH_MESSAGE,
        _MISSING_OPENAI_KEY_MESSAGE,
    ),
}


def test_direct_sessions_never_get_the_hf_upsell(run_ts):
    cases = [
        (model_id, message)
        for model_id in _picker_ids()
        if direct_provider_for(model_id) in _PROVIDER_MESSAGES
        for message in _PROVIDER_MESSAGES[direct_provider_for(model_id)]
    ]

    results = run_ts(
        [
            ("B.creditCtaForModel", [model_id, message, "free"])
            for model_id, message in cases
        ]
    )

    assert [r["kind"] for r in results] == [direct_provider_for(m) for m, _ in cases]


@pytest.mark.parametrize("provider", sorted(_PROVIDER_MESSAGES))
def test_provider_errors_keep_their_notice_after_switching_to_an_hf_model(
    run_ts, provider
):
    # The chat error banner re-renders with the session's current model.
    cases = [(m, msg) for m in HF_MODELS for msg in _PROVIDER_MESSAGES[provider]]

    results = run_ts(
        [
            ("B.creditCtaForModel", [model_id, message, "free"])
            for model_id, message in cases
        ]
    )

    assert [r["kind"] for r in results] == [provider] * len(cases)
    assert all("huggingface.co" not in json.dumps(r) for r in results)


def test_hf_credit_errors_keep_the_hf_cta(run_ts):
    results = run_ts(
        [
            ("B.creditCtaForModel", [HF_MODELS[0], "402 Payment Required", "free"]),
            ("B.creditCtaForModel", [HF_MODELS[1], "insufficient credits", "pro"]),
            # Raw provider phrases alone are not the backend's direct-provider
            # wording: an HF Router 429 keeps its usual handling.
            ("B.creditCtaForModel", [HF_MODELS[1], "429 insufficient_quota", "free"]),
            ("B.creditCtaForModel", [HF_MODELS[0], "invalid_api_key", "free"]),
        ]
    )

    assert [r and r["kind"] for r in results] == ["hf", "hf", None, None]
