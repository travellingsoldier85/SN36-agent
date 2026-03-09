from __future__ import annotations

from typing import Any, Dict, List, Optional

import os

import httpx


def is_sandbox_gateway_base_url(base_url: str) -> bool:
    """Detect whether requests are routed through the validator's local sandbox gateway.

    In the subnet runtime, the validator injects:
      OPENAI_BASE_URL=http://sandbox-gateway:9000/openai/v1

    and the gateway uses its own upstream provider keys. Agent code must not
    require (nor receive) real provider API keys in this mode.
    """
    if os.getenv("SANDBOX_GATEWAY_URL"):
        return True
    try:
        url = httpx.URL((base_url or "").strip())
        host = (url.host or "").lower()
        return host in {"sandbox-gateway", "localhost", "127.0.0.1"}
    except Exception:
        return False


def _llm_provider() -> str:
    return (os.getenv('LLM_PROVIDER') or os.getenv('OPENAI_PROVIDER') or 'openai').strip().lower()


class OpenAIGateway:
    """OpenAI-compatible /chat/completions client that injects IWA headers."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key  # If None, read from env per request.
        self.timeout_seconds = float(timeout_seconds)

    def chat_completions(self, *, task_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        api_key = self.api_key if self.api_key is not None else os.getenv("OPENAI_API_KEY", "")
        if not api_key and not is_sandbox_gateway_base_url(self.base_url):
            raise RuntimeError("OPENAI_API_KEY not set")

        headers = {
            "Content-Type": "application/json",
            "IWA-Task-ID": str(task_id),
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                json=body,
                headers=headers,
            )
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                detail = ""
                try:
                    j = e.response.json()
                    err = j.get("error", {}) if isinstance(j, dict) else {}
                    msg = str(err.get("message") or "")
                    msg = msg.replace("\n", " ").strip()
                    if len(msg) > 160:
                        msg = msg[:160] + "..."
                    detail = f"type={err.get('type')} code={err.get('code')} message={msg}"
                except Exception:
                    detail = (e.response.text or "")[:200]
                raise RuntimeError(f"OpenAI error ({e.response.status_code}): {detail}") from e

            return resp.json()


class AnthropicGateway:
    """Anthropic Messages API client that injects IWA headers and normalizes output.

    We normalize Anthropic's response into an OpenAI-like shape so the agent code can
    stay provider-agnostic.

    Docs:
    - POST https://api.anthropic.com/v1/messages
    - Headers: x-api-key, anthropic-version
    - Response includes usage: {input_tokens, output_tokens}
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: float = 30.0,
        anthropic_version: str = '2023-06-01',
    ) -> None:
        self.base_url = (base_url or os.getenv('ANTHROPIC_BASE_URL', 'https://api.anthropic.com')).rstrip('/')
        self.api_key = api_key  # If None, read from env per request.
        self.timeout_seconds = float(timeout_seconds)
        self.anthropic_version = anthropic_version

    def messages(self, *, task_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        api_key = self.api_key if self.api_key is not None else os.getenv('ANTHROPIC_API_KEY', '')
        if not api_key:
            raise RuntimeError('ANTHROPIC_API_KEY not set')

        headers = {
            'content-type': 'application/json',
            'x-api-key': api_key,
            'anthropic-version': self.anthropic_version,
            'IWA-Task-ID': str(task_id),
        }

        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.post(
                f"{self.base_url}/v1/messages",
                json=body,
                headers=headers,
            )
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                detail = (e.response.text or '')[:200]
                raise RuntimeError(f"Anthropic error ({e.response.status_code}): {detail}") from e

            raw = resp.json()

        # Extract text
        text = ''
        try:
            parts = raw.get('content') if isinstance(raw, dict) else None
            if isinstance(parts, list):
                bits = []
                for p in parts:
                    if isinstance(p, dict) and p.get('type') == 'text':
                        bits.append(str(p.get('text') or ''))
                text = ''.join(bits)
        except Exception:
            text = ''

        usage = raw.get('usage') if isinstance(raw, dict) else None
        input_tokens = int((usage or {}).get('input_tokens') or 0) if isinstance(usage, dict) else 0
        output_tokens = int((usage or {}).get('output_tokens') or 0) if isinstance(usage, dict) else 0

        # Normalize to OpenAI-like response.
        return {
            'choices': [{'message': {'content': text}}],
            'usage': {
                'prompt_tokens': input_tokens,
                'completion_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens,
            },
            'model': raw.get('model') if isinstance(raw, dict) else None,
        }


_openai = OpenAIGateway()
_chutes = OpenAIGateway(
    base_url=(os.getenv("CHUTES_BASE_URL") or "https://llm.chutes.ai/v1"),
    api_key=(os.getenv("CHUTES_API_KEY") or None),
)
_anthropic = AnthropicGateway()


def openai_chat_completions(
    *,
    task_id: str,
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 300,
) -> Dict[str, Any]:
    """Provider-agnostic "chat completions" used by the agent.

    For historical reasons this is named openai_chat_completions, but it supports:
    - LLM_PROVIDER=openai (OpenAI-compatible /chat/completions)
    - LLM_PROVIDER=anthropic (Anthropic Messages API, normalized output)

    Requirement for subnet gateway correlation:
    - Always inject header: IWA-Task-ID: <task_id>
    """

    provider = _llm_provider()
    m = str(model)

    if provider == "chutes":
        return _chutes.chat_completions(task_id=task_id, body={
            "model": m,
            "messages": messages,
            # Keep compatible defaults; some OpenAI-compatible providers may not
            # support response_format.
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        })

    if provider == 'anthropic':
        # Convert OpenAI-style messages into Anthropic messages+system.
        system = ''
        user = ''
        try:
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                if msg.get('role') == 'system' and not system:
                    system = str(msg.get('content') or '')
                if msg.get('role') == 'user' and not user:
                    user = str(msg.get('content') or '')
        except Exception:
            system = ''
            user = ''

        body: Dict[str, Any] = {
            'model': m,
            'max_tokens': int(max_tokens),
            'messages': [{'role': 'user', 'content': user}],
            'temperature': float(temperature),
        }
        if system:
            body['system'] = system

        return _anthropic.messages(task_id=task_id, body=body)

    # Default: OpenAI-compatible.
    body = {
        'model': m,
        'messages': messages,
    }

    # Model-specific parameterization.
    if m.startswith('gpt-5'):
        body['max_completion_tokens'] = int(max_tokens)
    else:
        body.update({
            'temperature': float(temperature),
            'max_tokens': int(max_tokens),
            'response_format': {'type': 'json_object'},
        })

    def _post(b: Dict[str, Any]) -> Dict[str, Any]:
        return _openai.chat_completions(task_id=task_id, body=b)

    try:
        return _post(body)
    except RuntimeError as e:
        msg = str(e)
        if 'unsupported_parameter' in msg or 'response_format' in msg:
            b2 = dict(body)
            b2.pop('response_format', None)
            return _post(b2)

        if 'unsupported_parameter' in msg and 'max_tokens' in body and 'max_completion_tokens' not in body:
            b2 = dict(body)
            b2.pop('max_tokens', None)
            b2.pop('temperature', None)
            b2['max_completion_tokens'] = int(max_tokens)
            b2.pop('response_format', None)
            return _post(b2)

        if 'unsupported_value' in msg and 'temperature' in body:
            b2 = dict(body)
            b2.pop('temperature', None)
            return _post(b2)

        raise
