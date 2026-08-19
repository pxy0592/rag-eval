import os
from typing import Literal, Optional

import torch
from dotenv import load_dotenv
from pydantic import BaseModel, computed_field

load_dotenv()


class Settings(BaseModel):
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    EMBEDDING_TOKEN_LIMIT: int = 8190
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "Qwen/Qwen3-4B-FP8")
    DTYPE: str = os.getenv("DTYPE", "float16")
    CTX_WINDOW: int = int(os.getenv("CTX_WINDOW", "8192"))
    TORCH_DEVICE: Optional[Literal["cuda", "cpu"]] = None
    ENVIRONMENT: str | Literal["dev", "prod"] = os.getenv("ENVIRONMENT", "prod")
    CLIENT_URL: str = os.getenv("CLIENT_URL", "http://localhost:8000/v1")

    @computed_field
    @property
    def DEVICE(self) -> Literal["cuda", "cpu"]:
        if self.TORCH_DEVICE:
            return self.TORCH_DEVICE
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"


settings = Settings()
