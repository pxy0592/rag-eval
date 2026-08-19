from typing import Generator

from ..lib.models.llm import OpenAIClient, vLLMClient
from ..lib.types import ChatMessage, RetrievedChunk
from ..settings import settings


def compress_history(
    history: list[ChatMessage], max_tokens: int = settings.CTX_WINDOW
) -> list[ChatMessage]:
    # TODO: Enhance to keep only relevant previous messages
    relevant_history = []
    total_tokens = 0
    for h in reversed(history):
        h_tokens = len(h["content"]) // 4
        if h_tokens + total_tokens > max_tokens:
            break
        relevant_history.append(h)
        total_tokens += h_tokens
    return list(reversed(relevant_history))


def compress_context(
    chunks: list[RetrievedChunk], max_tokens: int = settings.CTX_WINDOW
) -> str:
    # TODO: Use LLM to refine context instead of simple truncation
    formatted = [
        f"[Doc {i + 1}] {chunk.chunk.text}" for i, chunk in enumerate(chunks)
    ]
    context = "\n\n".join(formatted)
    return context[: max_tokens * 4]


def create_prompt(query: str, history: list[ChatMessage], context: str) -> str:
    prompt = ""
    for msg in history:
        if msg["role"] == "system" or msg["role"] == "assistant":
            prompt += f"<|Assistant|> {msg['content']}\n\n"
        elif msg["role"] == "user":
            prompt += f"<|User|> {msg['content']}\n\n"

    prompt += f"<|User|> {query}\n\n"

    if context:
        prompt += (
            f"<|Assistant|> Given the context: {context},"
            f"let's reason step-by-step about how to respond to your query: {query}."
            " <think>\n"
        )
    else:
        # TODO: logic to avoid hallucinations
        prompt += "<|Assistant|> <think>\n"

    # 'Provide step-by-step reasoning enclosed in <think> </think> tags, followed by the final answer enclosed in \boxed{} tags.' \
    # If its math
    # "Please reason step by step, and put your final answer within \boxed{}"
    return prompt


def generate_answer(
    query: str,
    history: list[ChatMessage] | None,
    chunks: list[RetrievedChunk] | None,
    model: vLLMClient | OpenAIClient,
    params: dict = {},
) -> Generator[str, None, None]:
    history = compress_history(history) if history else []
    context = compress_context(chunks) if chunks else ""
    prompt = create_prompt(
        query=query,
        history=history,
        context=context,
    )

    yield from model.generate_stream(prompt, params)
