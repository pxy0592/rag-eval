import os
from typing import Literal

import torch
from dotenv import load_dotenv
from pydantic import BaseModel, computed_field

load_dotenv()


class Settings(BaseModel):
    LLM_MODEL: str = os.getenv(
        "LLM_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    )
    DTYPE: str = os.getenv("DTYPE", "float16")
    CTX_WINDOW: int = int(os.getenv("CTX_WINDOW", "2048"))
    TORCH_DEVICE: Literal["cuda", "cpu"] | None = None
    ENVIRONMENT: str | Literal["dev", "prod"] = os.getenv("ENVIRONMENT", "prod")
    CLIENT_URL: str = os.getenv("CLIENT_URL", "http://localhost:8000/v1")
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    SMARTQ_API_URL: str | None = os.getenv("SMARTQ_API_URL")
    SMARTQ_API_KEY: str | None = os.getenv("SMARTQ_API_KEY")
    SMARTQ_TENANT_ID: str | None = os.getenv("SMARTQ_TENANT_ID")
    SMARTQ_AGENT_ID: str | None = os.getenv("SMARTQ_AGENT_ID")
    SMARTQ_KNOWLEDGE_BASE_IDS: str = os.getenv("SMARTQ_KNOWLEDGE_BASE_IDS", "")
    SMARTQ_KNOWLEDGE_IDS: str = os.getenv("SMARTQ_KNOWLEDGE_IDS", "")
    SMARTQ_AGENT_TIMEOUT_SECONDS: int = int(
        os.getenv("SMARTQ_AGENT_TIMEOUT_SECONDS", "180")
    )

    @computed_field
    @property
    def DEVICE(self) -> Literal["cuda", "cpu"]:
        if self.TORCH_DEVICE:
            return self.TORCH_DEVICE
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"


settings = Settings()
