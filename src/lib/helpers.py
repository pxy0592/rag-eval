import logging
from functools import lru_cache

import tiktoken

from lib.types import ChatMessage, Message

log = logging.getLogger("app")


@lru_cache(1000)
def count_tokens(text: str) -> int:
    if not text:
        return 0

    before = count_tokens.cache_info()
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text)
    count = len(tokens)
    after = count_tokens.cache_info()

    log.debug("input length=%d → tokens=%d", len(text), count)
    log.debug("cache hit=%s", after.hits > before.hits)
    return count


def extract_message_content(
    message: Message | str,
) -> tuple[str | None, list[str]]:
    log.info("extract_message_content(message=%s)", message)
    assert isinstance(message, dict) or isinstance(message, str)

    if isinstance(message, dict):
        return message.get("text", ""), message.get("files", [])
    return message, []


def parse_history(history: list[ChatMessage]) -> list[ChatMessage]:
    log.info("%d messages in", len(history))
    assert isinstance(history, list)
    assert all(isinstance(h, dict) for h in history)
    # TODO: this step only need to be run in the last history input
    # TODO: history could be empty
    # this is because gradio can make a chat content a tuple instead of a str
    # Ensure all the content is strings
    for h in history:
        if isinstance(h["content"], tuple):
            h["content"] = h["content"][1] if len(h["content"]) > 1 else ""
    return history
