"""Model construction for the environment roles and the target responder."""

from __future__ import annotations

import os

from inspect_ai.model import ChatMessageSystem, GenerateConfig, Model, get_model

RESPONDER_API_KEY_ENV = "RESPONDER_API_KEY"


def environment_model(
    name: str,
    *,
    reasoning_effort: str | None = None,
    max_tokens: int | None = None,
    attempt_timeout: int | None = None,
    max_retries: int | None = None,
) -> Model:
    """Model for the planner, user simulator, monitor, judge and seed audit."""
    config = {
        k: v for k, v in {
            "reasoning_effort": reasoning_effort,
            "max_tokens": max_tokens,
            "attempt_timeout": attempt_timeout,
            "max_retries": max_retries,
        }.items() if v is not None
    }
    return get_model(name, config=GenerateConfig(**config))


def responder_model(
    name: str,
    *,
    base_url: str | None = None,
    disable_thinking: bool = False,
    max_tokens: int | None = None,
) -> Model:
    """The target responder.

    With ``base_url`` the model is served from an OpenAI-compatible endpoint
    (a local vLLM server or a hosted provider such as OpenRouter); its key is
    read from ``RESPONDER_API_KEY`` at call time and never stored. With
    ``disable_thinking`` the request asks a Qwen3-style chat template to skip
    its reasoning block.
    """
    config: dict = {}
    if max_tokens is not None:
        config["max_tokens"] = max_tokens
    if base_url is None:
        return get_model(name, config=GenerateConfig(**config))
    if disable_thinking:
        config["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return get_model(
        name if name.startswith("openai/") else f"openai/{name}",
        base_url=base_url,
        api_key=os.environ.get(RESPONDER_API_KEY_ENV, "EMPTY"),
        responses_api=False,
        config=GenerateConfig(**config),
    )


def append_no_think(model: Model) -> None:
    """Append Qwen3's prompt-level ``/no_think`` switch to every request.

    Hosted providers may not forward ``chat_template_kwargs``, so this keeps
    hosted and local runs in the same non-thinking condition. Only the request
    changes; stored transcripts are untouched.
    """
    api = model.api
    generate = api.generate

    async def generate_no_think(input, tools, tool_choice, config):
        messages = list(input)
        idx = next((i for i, m in enumerate(messages) if m.role == "system"), None)
        if idx is not None and isinstance(messages[idx].content, str):
            messages[idx] = messages[idx].model_copy(
                update={"content": f"{messages[idx].content.rstrip()}\n\n/no_think"}
            )
        else:
            messages.insert(0 if idx is None else idx + 1, ChatMessageSystem(content="/no_think"))
        return await generate(input=messages, tools=tools, tool_choice=tool_choice, config=config)

    api.generate = generate_no_think


def model_name(model: str | Model | None) -> str | None:
    if model is None or isinstance(model, str):
        return model
    return getattr(model, "name", str(model))
