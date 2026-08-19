import torch
import vllm

from lib.settings import settings


class RerankerModel:
    def __init__(self):
        self._model = vllm.LLM(model=settings.RERANKER_MODEL, task="score")

    def compute_score(self, pairs: list[tuple[str, str]]) -> torch.Tensor:
        assert all(
            isinstance(pair, tuple) and len(pair) == 2 for pair in pairs
        ), f"Pairs should be list of tuples of size 2: {pairs}"

        scores = [
            self._model.score(text1, text2)[0].outputs.score
            for text1, text2 in pairs
        ]
        scores_tensor = torch.Tensor(scores)
        # TODO: make sure they are normalized
        return scores_tensor
