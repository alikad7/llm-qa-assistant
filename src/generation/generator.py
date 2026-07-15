from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from typing import Any

from src.generation.context_builder import BuiltContext
from src.generation.prompts import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    build_rag_messages,
)


class AnswerGenerationError(RuntimeError):
    pass


class AnswerGenerator:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 700,
        timeout_seconds: int = 60,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")

        if not base_url:
            raise ValueError("base_url is required")

        if not model:
            raise ValueError("model is required")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    def generate_answer(
        self,
        *,
        question: str,
        built_context: BuiltContext,
        answer_language: str = "en",
    ) -> str:
        question = question.strip()

        if not question:
            raise ValueError("question cannot be empty")

        if not built_context.context_text.strip():
            return self._empty_context_answer()

        messages = build_rag_messages(
            question=question,
            context=built_context.context_text,
        )

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        response_data = self._post_chat_completion(payload)
        answer = self._extract_answer(response_data)

        if not answer:
            raise AnswerGenerationError("LLM returned an empty answer")

        return answer.strip()

    def _post_chat_completion(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"

        body = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            url=url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                response_body = response.read().decode("utf-8")

        except TimeoutError as exc:
            raise AnswerGenerationError(
                "LLM API request timed out while waiting for a response. "
                "Try again, increase GENERATION_TIMEOUT_SECONDS, "
                "or reduce CONTEXT_MAX_CHARS / RETRIEVAL_TOP_K."
            ) from exc

        except socket.timeout as exc:
            raise AnswerGenerationError(
                "LLM API request timed out at the socket level. "
                "Try again, increase GENERATION_TIMEOUT_SECONDS, "
                "or reduce the amount of retrieved context."
            ) from exc

        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise AnswerGenerationError(
                f"LLM API HTTP error {exc.code}: {error_body}"
            ) from exc

        except urllib.error.URLError as exc:
            raise AnswerGenerationError(
                f"LLM API connection error: {exc.reason}"
            ) from exc

        try:
            return json.loads(response_body)

        except json.JSONDecodeError as exc:
            raise AnswerGenerationError(
                "LLM API returned invalid JSON"
            ) from exc

    @staticmethod
    def _extract_answer(
        response_data: dict[str, Any],
    ) -> str:
        choices = response_data.get("choices") or []

        if not choices:
            return ""

        first_choice = choices[0] or {}
        message = first_choice.get("message") or {}

        content = message.get("content")

        if isinstance(content, str):
            return content

        return ""

    @staticmethod
    def _empty_context_answer() -> str:
        return INSUFFICIENT_CONTEXT_MESSAGE
