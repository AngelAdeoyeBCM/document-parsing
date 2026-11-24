from __future__ import annotations

import base64
import json
import os
import urllib.request
from collections.abc import Sequence
from urllib.parse import urlparse


class VlmClient:
    """Minimal client for an OpenAI-compatible chat completions endpoint, e.g. vLLM serving a vision model."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str = 'VLM_API_KEY',
        timeout: float = 180.0,
        max_tokens: int = 512,
    ) -> None:
        if urlparse(base_url).scheme not in ('http', 'https'):
            raise ValueError(f'base_url must be http(s), got {base_url!r}')
        self.url = base_url.rstrip('/') + '/chat/completions'
        self.model = model
        self.api_key = os.environ.get(api_key_env)
        self.timeout = timeout
        self.max_tokens = max_tokens

    def ask(self, prompt: str, images: Sequence[bytes] = ()) -> str:
        """Send one prompt with PNG images and return the reply text."""
        content: list[dict[str, object]] = [{'type': 'text', 'text': prompt}]
        content.extend(
            {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64.b64encode(image).decode()}'}}
            for image in images
        )
        body = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': content}],
            'max_tokens': self.max_tokens,
            'temperature': 0,
        }
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        request = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers, method='POST')
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        return payload['choices'][0]['message']['content'] or ''
