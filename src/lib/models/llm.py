import logging
from typing import Generator, Type

import vllm
from openai import OpenAI
from pydantic import BaseModel
from vllm.sampling_params import GuidedDecodingParams

from lib.helpers import count_tokens
from lib.settings import settings
from lib.types import ChatMessage, GenerationParams

DEFAULT_PARAMS: GenerationParams = {
    "max_tokens": 500,
    "temperature": 0.25,
    "top_p": 0.95,
    "frequency_penalty": 0.5,
    "presence_penalty": 1.2,
}

log = logging.getLogger("app")


class OpenAIClient:
    def __init__(self) -> None:
        log.info("llm: starting client in dev mode")
        self.model = OpenAI(
            api_key="API",
            base_url=settings.CLIENT_URL,
        )

    def generate(
        self,
        prompt: str,
        output_format: Type[BaseModel] | None,
        params: GenerationParams = {},
    ) -> str:
        assert count_tokens(prompt) < settings.CTX_WINDOW
        assert prompt

        params = DEFAULT_PARAMS | params
        response = self.model.chat.completions.create(
            messages=[{"role": "assistant", "content": prompt}],
            model=settings.LLM_MODEL,
            extra_body={"guided_json": output_format.model_json_schema()}
            if output_format
            else None,
            **params,
        )
        return response.choices[0].message.content or ""

    def generate_stream(
        self,
        messages: list[ChatMessage],
        params: GenerationParams = {},
    ) -> Generator[str, None, None]:
        assert isinstance(messages, list)

        params = DEFAULT_PARAMS | params
        response = self.model.chat.completions.create(
            messages=messages,  # type: ignore
            model=settings.LLM_MODEL,
            stream=True,
            **params,
        )
        for output in response:
            yield output.choices[0].delta.content or ""


class vLLMClient:
    def __init__(self) -> None:
        log.info(
            "llm: starting vllm with:\nmodel:%s - dtype:%s - max_model_len:%s",
            settings.LLM_MODEL,
            settings.DTYPE,
            settings.CTX_WINDOW,
        )
        self.model = vllm.LLM(
            model=settings.LLM_MODEL,
            dtype=settings.DTYPE,
            task="generate",
            enforce_eager=False,
            max_model_len=settings.CTX_WINDOW,
            max_num_seqs=2,
            enable_chunked_prefill=True,
            gpu_memory_utilization=0.6,
            reasoning_parser="deepseek_r1",
            guided_decoding_backend="xgrammar"
        )

    def generate(
        self,
        prompt: str,
        output_format: Type[BaseModel] | None = None,
        params: GenerationParams = {},
    ) -> str:
        assert count_tokens(prompt) < settings.CTX_WINDOW
        assert prompt

        params = DEFAULT_PARAMS | params
        response = self.model.generate(
            prompt,
            sampling_params=vllm.SamplingParams(
                **params,
                guided_decoding=GuidedDecodingParams(
                    json=output_format.model_json_schema()
                )
                if output_format
                else None,
            ),
        )
        return response[0].outputs[0].text

    def generate_stream(
        self,
        messages: list[ChatMessage],
        params: GenerationParams = {},
    ) -> Generator[str, None, None]:
        assert isinstance(messages, list)

        params = DEFAULT_PARAMS | params
        response = self.model.generate(
            [messages], sampling_params=vllm.SamplingParams(**params)
        )
        for output in response:
            yield output.outputs[0].text
