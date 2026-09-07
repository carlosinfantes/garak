"""OpenRouter LLM interface

Connects to OpenRouter's unified, OpenAI-compatible endpoint, giving access
to hundreds of models hosted by many different providers behind a single
API key.
"""

import functools
import logging
from typing import List, Union

import openai

from garak.attempt import Message, Conversation
from garak.exception import BadGeneratorException
from garak.generators.openai import OpenAICompatible


class OpenRouterGenerator(OpenAICompatible):
    """Wrapper for OpenRouter (https://openrouter.ai), a unified API giving
    access to hundreds of models from many different providers.

    Expects the ``OPENROUTER_API_KEY`` environment variable to be set to an
    OpenRouter API key; see https://openrouter.ai/keys for details on
    creating one.

    Model names follow OpenRouter's ``<provider>/<model>`` convention, e.g.
    ``anthropic/claude-sonnet-4``. Browse available models at
    https://openrouter.ai/models.
    """

    ENV_VAR = "OPENROUTER_API_KEY"
    DEFAULT_PARAMS = OpenAICompatible.DEFAULT_PARAMS | {
        "uri": "https://openrouter.ai/api/v1",
        # per https://openrouter.ai/docs/api-reference/parameters and
        # github.com/OpenRouterTeam/openrouter-runner issue 99, `n > 1` is not
        # supported: OpenRouter fans a request out to a single upstream
        # provider per call, so multiple completions require multiple requests.
        "suppressed_params": {"n"},
    }
    active = True
    supports_multiple_generations = False
    generator_family_name = "OpenRouter"

    @staticmethod
    def _reject_insufficient_credit(create_fn):
        """Wrap an OpenAI-SDK ``create`` callable so that OpenRouter's HTTP 402
        (out of credit) is treated as terminal instead of being logged and
        skipped like other non-transient errors. Retrying a 402 never helps,
        and silently returning ``None`` for every prompt would let a long scan
        run to completion producing nothing but empty results.

        ``functools.wraps`` preserves ``create_fn``'s signature via
        ``__wrapped__`` so that ``OpenAICompatible._call_model``, which
        inspects ``generator.create``'s parameters to build the request, keeps
        seeing the real argument names (``model``, ``messages``, ``n``, ...)
        instead of this wrapper's generic ``*args, **kwargs``.
        """

        @functools.wraps(create_fn)
        def wrapper(*args, **kwargs):
            try:
                return create_fn(*args, **kwargs)
            except openai.APIStatusError as e:
                if e.status_code == 402:
                    msg = (
                        "OpenRouter account has insufficient credit (HTTP 402); "
                        "top up at https://openrouter.ai/credits before retrying."
                    )
                    logging.error(msg)
                    # raised from None: openai.APIStatusError carries a non-picklable
                    # httpx.Response, which crashes multiprocessing.Pool's result
                    # handling if attached as __cause__ (see NVIDIA/garak#1357).
                    raise BadGeneratorException(msg) from None
                raise

        return wrapper

    def _load_unsafe(self):
        if self.name in ("", None):
            raise ValueError(
                "OpenRouter requires model name to be set, e.g. "
                "--target_name anthropic/claude-sonnet-4\n"
                "Browse available models at https://openrouter.ai/models"
            )
        super()._load_unsafe()
        self.generator.create = self._reject_insufficient_credit(self.generator.create)

    def _call_model(
        self, prompt: Conversation, generations_this_call: int = 1
    ) -> List[Union[Message, None]]:
        if generations_this_call != 1:
            raise AssertionError("generations_per_call / n > 1 is not supported")
        return super()._call_model(prompt, generations_this_call)


DEFAULT_CLASS = "OpenRouterGenerator"
