import json
from pathlib import Path

from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import FactualCorrectness, Faithfulness, LLMContextRecall
from lib.settings import settings

from metrics import (
    calc_map,
    calc_mrr,
    calc_ndcg,
    calc_precision,
    calc_recall,
)


# Configuration ----------------------------------------------------------------

RESULTS_PATH = Path(__file__).parent / "data" / "results.jsonl"
CUTOFFS = [1, 5, 10]
MAX_K = max(CUTOFFS)

evaluator_llm = LangchainLLMWrapper(
    ChatOpenAI(
        api_key="NULL", base_url=settings.CLIENT_URL, model=settings.LLM_MODEL
    )
)


# Evaluate ---------------------------------------------------------------------


def main() -> None:
    evaluation_dataset = EvaluationDataset.from_jsonl(RESULTS_PATH)

    # 1. Eval Retrieval
    # retrieved_ids = [item.retrieved_ids for item in evaluation_dataset]
    # ground_truth = [item.ground_truth_ids for item in evaluation_dataset]

    # PRECISIONs = calc_precision(retrieved_ids, ground_truth, CUTOFFS)
    # RECALLs = calc_recall(retrieved_ids, ground_truth, CUTOFFS)
    # MRRs = calc_mrr(retrieved_ids, ground_truth, CUTOFFS)
    # NDCGs = calc_ndcg(retrieved_ids, ground_truth, CUTOFFS)
    # MAPs = calc_map(retrieved_ids, ground_truth, CUTOFFS)

    # retrieval_results.append({
    #     "query": item.user_input,
    #     "retrieval_metrics": {
    #         **{f"precision@{c}": PRECISIONs[i] for i, c in enumerate(CUTOFFS)},
    #         **{f"recall@{c}": RECALLs[i] for i, c in enumerate(CUTOFFS)},
    #         **{f"mrr@{c}": MRRs[i] for i, c in enumerate(CUTOFFS)},
    #         **{f"ndcg@{c}": NDCGs[i] for i, c in enumerate(CUTOFFS)},
    #         **{f"map@{c}": MAPs[i] for i, c in enumerate(CUTOFFS)},
    #     },
    # })

    # 2. Eval Generation
    generation_eval_result = evaluate(
        dataset=evaluation_dataset,
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
        llm=evaluator_llm,
    )

    print(generation_eval_result)


if __name__ == "__main__":
    main()
