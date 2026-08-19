from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.lib import llm as llm_module


class FakeOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=Mock()))


class FakeLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate(self, prompt, sampling_params):
        return [SimpleNamespace(outputs=[SimpleNamespace(text='{"question":"Q","answer":"A"}')])]


def test_get_client_dev(monkeypatch):
    monkeypatch.setattr(llm_module, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(llm_module.settings, "ENVIRONMENT", "dev")

    client = llm_module.get_client()

    assert isinstance(client, FakeOpenAI)
    assert client.kwargs == {"api_key": "API", "base_url": llm_module.settings.CLIENT_URL}


def test_get_client_prod(monkeypatch):
    monkeypatch.setattr(llm_module, "LLM", FakeLLM)
    monkeypatch.setattr(llm_module.settings, "ENVIRONMENT", "prod")

    client = llm_module.get_client()

    assert isinstance(client, FakeLLM)
    assert client.kwargs["model"] == llm_module.settings.LLM_MODEL


def test_get_client_rejects_unknown_environment(monkeypatch):
    monkeypatch.setattr(llm_module.settings, "ENVIRONMENT", "test")

    with pytest.raises(ValueError, match="env should be dev \\| prod"):
        llm_module.get_client()


def test_generate_dev(monkeypatch):
    client = FakeOpenAI()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"question":"Q","answer":"A"}'))]
    )
    monkeypatch.setattr(llm_module, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(llm_module.settings, "ENVIRONMENT", "dev")

    result = llm_module.generate("prompt", client, {"max_tokens": 12})

    assert result.question == "Q"
    client.chat.completions.create.assert_called_once()
    assert client.chat.completions.create.call_args.kwargs["max_tokens"] == 12


def test_generate_prod(monkeypatch):
    monkeypatch.setattr(llm_module, "LLM", FakeLLM)
    monkeypatch.setattr(llm_module.settings, "ENVIRONMENT", "prod")

    result = llm_module.generate("prompt", FakeLLM())

    assert result.answer == "A"


def test_generate_rejects_invalid_inputs(monkeypatch):
    monkeypatch.setattr(llm_module.settings, "ENVIRONMENT", "dev")
    with pytest.raises(AssertionError, match="OpenAI"):
        llm_module.generate("prompt", object())

    monkeypatch.setattr(llm_module.settings, "ENVIRONMENT", "test")
    with pytest.raises(ValueError, match="Correctly configure"):
        llm_module.generate("prompt", object())


def test_generate_rejects_empty_dev_response(monkeypatch):
    client = FakeOpenAI()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
    )
    monkeypatch.setattr(llm_module, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(llm_module.settings, "ENVIRONMENT", "dev")

    with pytest.raises(RuntimeError, match="Something appened"):
        llm_module.generate("prompt", client)


def test_generate_rejects_oversized_prompt(monkeypatch):
    monkeypatch.setattr(llm_module.settings, "CTX_WINDOW", 1)

    with pytest.raises(AssertionError, match="Prompt too large"):
        llm_module.generate("12345", object())
