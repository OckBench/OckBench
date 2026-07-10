"""Common base interface for all provider clients.

Every provider builds the same way: ``build_request`` assembles the full
provider-specific request dict, the shared ``shape_request`` applies the user's
``request_overrides`` (with ``${max_output_tokens}`` substitution), and
``_dispatch`` sends the shaped request and parses the response. ``generate``
here is the single owner of retry/backoff and non-retryable classification — no
provider implements its own retry loop, and every provider client disables its
underlying SDK's auto-retry so attempts never stack.

Connection parameters (``timeout``, ``max_retries``, ``base_url``, ``api_key``)
and the optional ``wall_clock_timeout``) are constructor arguments; generation
parameters that stay constant across a run (``temperature``, ``top_p``,
``request_overrides``) are held on the client; only the per-problem
``max_output_tokens`` is passed to ``generate``.
"""
import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from ..core.schemas import ModelResponse, RequestOverrides, TokenUsage
from ..utils.request_overrides import apply_request_overrides, guard_protected_paths, override_paths

logger = logging.getLogger(__name__)


def raise_status_error(status_code: int, body: str) -> None:
    """Raise a transport error carrying ``status_code`` for retry classification.

    The raw-httpx clients (anthropic, openai-responses) catch an
    ``HTTPStatusError`` and re-raise through here so ``_is_non_retryable_error``
    can read ``status_code`` off the exception. Always raises.
    """
    error = Exception(f"Error code: {status_code} - {body}")
    error.status_code = status_code
    raise error


class BaseModelClient(ABC):
    """Abstract base class for provider clients with the single retry owner."""

    #: Request fields this provider depends on for streaming / token accounting.
    #: A user override targeting any of these (or a path beneath them) is rejected.
    protected_paths: Tuple[str, ...] = ()
    #: Registered provider name; used in protected-path error messages.
    provider_name: str = "base"

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
        wall_clock_timeout: Optional[int] = None,
        max_retries: int = 3,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        request_overrides: Optional[RequestOverrides] = None,
        **kwargs,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.wall_clock_timeout = wall_clock_timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.top_p = top_p
        self.request_overrides = request_overrides if request_overrides is not None else RequestOverrides()
        self.extra_params = kwargs

        # Enforce this provider's protected paths once, at construction, before
        # any network — the error names the offending field and the provider.
        guard_protected_paths(
            override_paths(self.request_overrides),
            self.protected_paths,
            provider=self.provider_name,
        )

    # ----- provider-specific surface ------------------------------------- #
    @abstractmethod
    def build_request(self, prompt: str, max_output_tokens: int) -> Dict[str, Any]:
        """Assemble the full provider-specific request dict (pre-override)."""

    @abstractmethod
    async def _dispatch(self, request: Dict[str, Any]) -> ModelResponse:
        """Send the shaped request and parse it into a ``ModelResponse``.

        Must not implement retry; ``generate`` owns that. Must raise on a
        transport/HTTP failure so ``generate`` can classify and retry it.
        """

    def close(self) -> None:
        """Release any client-owned resources. No-op by default; overridden by
        clients that own an executor or connection pool."""

    async def aclose(self) -> None:
        """Release async resources before the runner's event loop exits."""
        self.close()

    # ----- shared request-shaping + retry -------------------------------- #
    def shape_request(self, prompt: str, max_output_tokens: int) -> Dict[str, Any]:
        """Build the base request then apply the user's request overrides.

        Pure (no network): also used by the inspect/dry-run surface.
        """
        request = self.build_request(prompt, max_output_tokens)
        return apply_request_overrides(
            request,
            self.request_overrides,
            {"max_output_tokens": max_output_tokens},
        )

    async def generate(self, prompt: str, max_output_tokens: int) -> ModelResponse:
        """Generate a response, owning the only retry/backoff loop."""
        request = self.shape_request(prompt, max_output_tokens)
        last_error: Optional[Exception] = None
        start_time = time.time()

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = await self._dispatch_with_wall_clock_timeout(request)
                response.latency = time.time() - start_time
                return response
            except asyncio.TimeoutError as e:
                if self.wall_clock_timeout is not None:
                    last_error = Exception(
                        f"wall_clock_timeout: attempt exceeded {self.wall_clock_timeout}s"
                    )
                else:
                    last_error = Exception(str(e) or type(e).__name__)
                error_msg = str(last_error)
            except Exception as e:  # noqa: BLE001 - classified below
                last_error = e
                error_msg = str(e)

                if self._is_non_retryable_error(e):
                    logger.error(f"[{self.provider_name}] non-retryable error: {error_msg}")
                    return self._create_error_response(error_msg, time.time() - start_time)

            if attempt < self.max_retries - 1:
                wait_time = self._retry_wait_time(error_msg, attempt)
                logger.warning(
                    f"[{self.provider_name}] API call failed "
                    f"(attempt {attempt + 1}/{self.max_retries}): {error_msg}. "
                    f"Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(
                    f"[{self.provider_name}] API call failed after "
                    f"{self.max_retries} attempts: {error_msg}"
                )

        return self._create_error_response(str(last_error), 0)

    async def _dispatch_with_wall_clock_timeout(self, request: Dict[str, Any]) -> ModelResponse:
        if self.wall_clock_timeout is None:
            return await self._dispatch(request)
        return await asyncio.wait_for(
            self._dispatch(request),
            timeout=float(self.wall_clock_timeout),
        )

    def _is_non_retryable_error(self, error: Exception) -> bool:
        """Classify errors that must surface immediately without retry."""
        status = getattr(error, "status_code", None)
        if status in (400, 401, 403, 404, 422):
            return True

        non_retryable_keywords = [
            "invalid api key",
            "authentication failed",
            "invalid model",
            "context length exceeded",
            "content policy violation",
            "invalid request",
        ]
        error_lower = str(error).lower()
        return any(keyword in error_lower for keyword in non_retryable_keywords)

    def _retry_wait_time(self, error_msg: str, attempt: int) -> float:
        """Back off longer for provider throttling and broken streaming bodies."""
        error_lower = error_msg.lower()
        retry_pressure_keywords = [
            "429", "rate limit", "too many requests", "peer closed",
            "incomplete message body", "remoteprotocolerror", "readtimeout", "timeout",
        ]
        if any(keyword in error_lower for keyword in retry_pressure_keywords):
            base_wait = min(10 * (2 ** attempt), 120)
        else:
            base_wait = min(2 ** attempt, 30)
        return round(base_wait + random.uniform(0, 1.5), 2)

    def _create_error_response(self, error_msg: str, latency: float) -> ModelResponse:
        return ModelResponse(
            text="",
            tokens=TokenUsage(
                prompt_tokens=0, answer_tokens=0, reasoning_tokens=0,
                output_tokens=0, total_tokens=0,
            ),
            latency=latency,
            model=self.model,
            error=error_msg,
            finish_reason="error",
        )
