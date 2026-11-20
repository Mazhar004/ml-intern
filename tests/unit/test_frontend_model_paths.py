"""The web picker's frontend model lists match the picker spec.

The frontend has no unit-test runner, so these tests read the TypeScript
sources: the direct-provider allowlists and billing captions in
``utils/model.ts``, the fallback picker list in ``Chat/ChatInput.tsx`` and the
OpenAI error phrases in ``utils/inferenceBilling.ts``. The expected lists are
hardcoded from the picker spec, which ``AVAILABLE_MODELS`` in
``agent/core/web_models.py`` follows too, and the allowlists are also checked
against ``agent/core/model_ids.py``. Comments are stripped before parsing, so
prose that mentions an id is never read as code.
"""

import re
from pathlib import Path

import pytest

_FRONTEND_SRC = Path(__file__).resolve().parents[2] / "frontend" / "src"
MODEL_TS = _FRONTEND_SRC / "utils" / "model.ts"
INFERENCE_BILLING_TS = _FRONTEND_SRC / "utils" / "inferenceBilling.ts"
CHAT_INPUT_TSX = _FRONTEND_SRC / "components" / "Chat" / "ChatInput.tsx"
APP_LAYOUT_TSX = _FRONTEND_SRC / "components" / "Layout" / "AppLayout.tsx"

HF_CREDITS = "HF credits"
ANTHROPIC_KEY = "Your Anthropic key"
OPENAI_KEY = "Your OpenAI key"
DEEPSEEK_KEY = "Your DeepSeek key"

# (picker id, label, billing caption) in display order.
PICKER_MODELS = [
    ("anthropic/claude-opus-4.8:fal-ai", "Claude Opus 4.8", HF_CREDITS),
    ("anthropic/claude-opus-5-5", "Claude Opus 5.5", ANTHROPIC_KEY),
    ("anthropic/claude-fable-5-2", "Claude Fable 5.2", ANTHROPIC_KEY),
    ("anthropic/claude-sonnet-5-5", "Claude Sonnet 5.5", ANTHROPIC_KEY),
    ("anthropic/claude-haiku-5-5", "Claude Haiku 5.5", ANTHROPIC_KEY),
    ("openai/gpt-5.5:fal-ai", "GPT-5.5", HF_CREDITS),
    ("openai/gpt-5.5", "GPT-5.5 (OpenAI key)", OPENAI_KEY),
    ("openai/gpt-6.2-astra", "GPT-6.2 Astra", OPENAI_KEY),
    ("openai/gpt-6.2-sol", "GPT-6.2 Sol", OPENAI_KEY),
    ("openai/gpt-6.1-terra", "GPT-6.1 Terra", OPENAI_KEY),
    ("openai/gpt-6-luna", "GPT-6 Luna", OPENAI_KEY),
    ("moonshotai/Kimi-K2.7-Code:novita", "Kimi K2.7 Code", HF_CREDITS),
    ("MiniMaxAI/MiniMax-M3:novita", "MiniMax M3", HF_CREDITS),
    ("zai-org/GLM-5.2:novita", "GLM 5.2", HF_CREDITS),
    ("deepseek/deepseek-v4-pro", "DeepSeek V4 Pro", DEEPSEEK_KEY),
    ("deepseek/deepseek-v4-flash", "DeepSeek V4 Flash", DEEPSEEK_KEY),
]
RECOMMENDED_ID = "anthropic/claude-opus-5-5"
DIRECT_ANTHROPIC_IDS = {
    "anthropic/claude-opus-5-5",
    "anthropic/claude-fable-5-2",
    "anthropic/claude-sonnet-5-5",
    "anthropic/claude-haiku-5-5",
}
DIRECT_OPENAI_IDS = {
    "openai/gpt-5.5",
    "openai/gpt-6.2-astra",
    "openai/gpt-6.2-sol",
    "openai/gpt-6.1-terra",
    "openai/gpt-6-luna",
}
BILLING_LABELS = {
    "hf": HF_CREDITS,
    "anthropic": ANTHROPIC_KEY,
    "openai": OPENAI_KEY,
    "deepseek": DEEPSEEK_KEY,
}
OPENAI_KEY_PHRASES = [
    "openai rejected the api key",
    "gpt via the openai api needs openai_api_key",
    "incorrect api key provided",
    "invalid_api_key",
]
OPENAI_QUOTA_PHRASES = [
    "your openai api quota is exhausted",
    "insufficient_quota",
    "exceeded your current quota",
]
OPENAI_NOTICE_TITLES = {
    "OPENAI_QUOTA_NOTICE": "OpenAI quota exhausted",
    "OPENAI_KEY_NOTICE": "OpenAI API key missing or invalid",
}

_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)
# A line comment starts the line or follows whitespace, which leaves URLs
# ("https://") and escaped slashes in regex literals ("\//") alone.
_LINE_COMMENT_RE = re.compile(r"(^|\s)//[^\n]*")
_STRING_CONST_RE = re.compile(r"\bexport const (\w+) = '([^']*)';")
_STRING_RE = re.compile(r"'([^']*)'")
_FORBIDDEN_ROUTING_RE = re.compile(
    r"""\.(?:startsWith\(\s*['"`](?:openai|anthropic)|includes\(\s*['"`]openai)"""
)


def _code(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return _LINE_COMMENT_RE.sub(r"\1", _BLOCK_COMMENT_RE.sub("", text))


def _declaration_body(code: str, name: str, opener: str, closer: str) -> str:
    """The text between ``opener`` and ``closer`` in ``const <name> ... =``."""
    pattern = rf"\bconst {name}\b[^=]*=\s*{re.escape(opener)}(.*?){re.escape(closer)}"
    match = re.search(pattern, code, re.S)
    assert match, f"const {name} = {opener}...{closer} not found"
    return match.group(1)


def _resolve(token: str, constants: dict[str, str]) -> str:
    """A quoted string literal, or a model.ts string constant by name."""
    token = token.strip()
    literal = _STRING_RE.fullmatch(token)
    if literal:
        return literal.group(1)
    assert token in constants, f"{token} is not a string constant in model.ts"
    return constants[token]


@pytest.fixture(scope="module")
def model_code() -> str:
    return _code(MODEL_TS)


@pytest.fixture(scope="module")
def constants(model_code: str) -> dict[str, str]:
    return dict(_STRING_CONST_RE.findall(model_code))


def _path_set(model_code: str, constants: dict[str, str], name: str) -> list[str]:
    body = _declaration_body(model_code, name, "new Set([", "])")
    tokens = [token for token in body.split(",") if token.strip()]
    return [_resolve(token, constants) for token in tokens]


@pytest.fixture(scope="module")
def default_model_options(constants: dict[str, str]) -> list[dict]:
    body = _declaration_body(_code(CHAT_INPUT_TSX), "DEFAULT_MODEL_OPTIONS", "[", "\n]")
    options = []
    for entry in re.findall(r"\{([^{}]*)\}", body):
        fields = dict(re.findall(r"\b(\w+):\s*('[^']*'|\w+)", entry))
        options.append(
            {
                "id": _resolve(fields["id"], constants),
                "name": _resolve(fields["name"], constants),
                "modelPath": _resolve(fields["modelPath"], constants),
                "recommended": fields.get("recommended") == "true",
            }
        )
    return options


def _billing_route(model_id: str, anthropic: set[str], openai: set[str]) -> str:
    """Python mirror of billingRouteForPath, fed with the parsed allowlists."""
    if model_id in anthropic:
        return "anthropic"
    if model_id in openai:
        return "openai"
    if model_id.startswith("deepseek/"):
        return "deepseek"
    return "hf"


def test_direct_openai_paths_are_the_five_openai_ids(model_code, constants):
    paths = _path_set(model_code, constants, "DIRECT_OPENAI_PATHS")

    assert sorted(paths) == sorted(DIRECT_OPENAI_IDS)
    assert "openai/gpt-5.5:fal-ai" not in paths


def test_direct_anthropic_paths_are_the_four_claude_ids(model_code, constants):
    paths = _path_set(model_code, constants, "DIRECT_ANTHROPIC_PATHS")

    assert sorted(paths) == sorted(DIRECT_ANTHROPIC_IDS)
    assert "anthropic/claude-opus-4.8:fal-ai" not in paths


def test_allowlists_match_backend_model_ids(model_code, constants):
    # The backend's exact-membership sets decide where a call is billed.
    from agent.core.model_ids import (
        DIRECT_ANTHROPIC_MODEL_IDS,
        DIRECT_OPENAI_MODEL_IDS,
    )

    anthropic = _path_set(model_code, constants, "DIRECT_ANTHROPIC_PATHS")
    openai = _path_set(model_code, constants, "DIRECT_OPENAI_PATHS")

    assert set(anthropic) == DIRECT_ANTHROPIC_MODEL_IDS
    assert set(openai) == DIRECT_OPENAI_MODEL_IDS


def test_default_model_options_follow_picker_order(default_model_options):
    paths = [option["modelPath"] for option in default_model_options]

    assert paths == [model_id for model_id, _, _ in PICKER_MODELS]


def test_default_model_options_use_picker_labels(default_model_options):
    names = [option["name"] for option in default_model_options]

    assert names == [label for _, label, _ in PICKER_MODELS]


def test_default_model_options_match_backend_available_models(default_model_options):
    # The fallback list, and the label shown for a hidden session model, must
    # match what /api/config/model serves.
    from agent.core.web_models import AVAILABLE_MODELS

    frontend = [
        (option["modelPath"], option["name"], option["recommended"])
        for option in default_model_options
    ]
    backend = [
        (model["id"], model["label"], model.get("recommended", False))
        for model in AVAILABLE_MODELS
    ]

    assert frontend == backend


def test_default_model_option_ids_are_unique(default_model_options):
    ids = [option["id"] for option in default_model_options]
    id_by_path = {option["modelPath"]: option["id"] for option in default_model_options}

    assert len(ids) == len(set(ids))
    assert id_by_path["openai/gpt-5.5:fal-ai"] == "gpt-5.5"
    assert id_by_path["openai/gpt-5.5"] == "gpt-5.5-openai"


def test_only_claude_opus_55_is_recommended(default_model_options):
    recommended = [
        option["modelPath"] for option in default_model_options if option["recommended"]
    ]

    assert recommended == [RECOMMENDED_ID]


def test_billing_labels_name_who_pays(model_code):
    body = _declaration_body(model_code, "BILLING_LABELS", "{", "}")

    assert dict(re.findall(r"(\w+):\s*'([^']*)'", body)) == BILLING_LABELS


def test_every_picker_model_gets_its_billing_caption(model_code, constants):
    anthropic = set(_path_set(model_code, constants, "DIRECT_ANTHROPIC_PATHS"))
    openai = set(_path_set(model_code, constants, "DIRECT_OPENAI_PATHS"))
    body = _declaration_body(model_code, "BILLING_LABELS", "{", "}")
    labels = dict(re.findall(r"(\w+):\s*'([^']*)'", body))

    # DeepSeek is the one prefix route, mirroring direct_provider_for.
    assert "startsWith('deepseek/')" in model_code
    captions = [
        labels[_billing_route(model_id, anthropic, openai)]
        for model_id, _, _ in PICKER_MODELS
    ]
    assert captions == [caption for _, _, caption in PICKER_MODELS]


@pytest.mark.parametrize(
    "path", [MODEL_TS, CHAT_INPUT_TSX, INFERENCE_BILLING_TS, APP_LAYOUT_TSX]
)
def test_routing_never_uses_openai_or_anthropic_prefixes(path):
    # Router ids share these prefixes ("openai/gpt-5.5:fal-ai"), so direct ids
    # are matched by exact membership only.
    assert _FORBIDDEN_ROUTING_RE.findall(_code(path)) == []


def test_inference_billing_matches_openai_phrases_verbatim():
    code = _code(INFERENCE_BILLING_TS)

    key_phrases = _STRING_RE.findall(
        _declaration_body(code, "OPENAI_KEY_PHRASES", "[", "]")
    )
    quota_phrases = _STRING_RE.findall(
        _declaration_body(code, "OPENAI_QUOTA_PHRASES", "[", "]")
    )
    assert key_phrases == OPENAI_KEY_PHRASES
    assert quota_phrases == OPENAI_QUOTA_PHRASES
    # Errors are lowercased before matching, so a capital would never match.
    assert all(phrase == phrase.lower() for phrase in key_phrases + quota_phrases)


@pytest.mark.parametrize("name, title", OPENAI_NOTICE_TITLES.items())
def test_openai_notices_have_no_hf_links(name, title):
    notice = _declaration_body(_code(INFERENCE_BILLING_TS), name, "{", "};")
    fields = dict(re.findall(r"(\w+):\s*'([^']*)'", notice))

    assert fields["kind"] == "openai"
    assert fields["title"] == title
    assert "huggingface" not in notice.lower()
    assert "href" not in notice.lower()
