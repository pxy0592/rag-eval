from openai import OpenAI
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from .types import QAFormat
from ..settings import settings


def get_client() -> LLM | OpenAI:
    match settings.ENVIRONMENT:
        case "dev":
            model = OpenAI(api_key="API", base_url=settings.CLIENT_URL)
        case "prod":
            model = LLM(
                model=settings.LLM_MODEL,
                dtype=settings.DTYPE,
                task="generate",
                enforce_eager=False,
                max_model_len=settings.CTX_WINDOW,
                max_num_seqs=2,
                enable_chunked_prefill=True,
            )
        case _:
            raise ValueError("env should be dev | prod")
    return model


def generate(prompt: str, llm: LLM | OpenAI, params: dict = {}) -> QAFormat:
    assert len(prompt) // 4 < settings.CTX_WINDOW, "Prompt too large!!"

    max_tokens = params.get("max_tokens", 100)
    temperature = params.get("temperature", 0.25)
    top_p = params.get("top_p", 0.95)
    frequency_penalty = params.get("frequency_penalty", 0.5)
    presence_penalty = params.get("presence_penalty", 1.2)
    repetition_penalty = params.get("repetition_penalty", 1.2)

    response = ""
    match settings.ENVIRONMENT:
        case "prod":
            assert isinstance(llm, LLM), (
                "LLM should be an instance of LLM class"
            )
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                structured_outputs=StructuredOutputsParams(
                    json=QAFormat.model_json_schema()
                ),
            )
            response = llm.generate(prompt, sampling_params)[0].outputs[0].text
        case "dev":
            assert isinstance(llm, OpenAI), (
                "LLM should be and instance of OpenAI class"
            )
            response = (
                llm.chat.completions.create(
                    messages=[{"role": "assistant", "content": prompt}],
                    model=settings.LLM_MODEL,
                    extra_body={"guided_json": QAFormat.model_json_schema()},
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                )
                .choices[0]
                .message.content
            )
            if not response:
                raise RuntimeError("Something appened while trying to generate")
        case _:
            raise ValueError("Correctly configure the env to be dev | prod")

    return QAFormat.model_validate_json(response)
