import json
from pathlib import Path

from utils import DatasetDocument, DatasetQA, EvaluationDataset

from core.generation import generate_answer
from lib.models.embedding import EmbeddingModel
from lib.models.llm import OpenAIClient
from lib.models.rerank import RerankerModel
from lib.vectordb import KnowledgeBase

# Configuration ----------------------------------------------------------------


DATA_PATH = Path(__file__).parent / "data"
DOCS_PATH = DATA_PATH / "wiki_docs.jsonl"
QA_PATH = DATA_PATH / "wiki_qa.jsonl"
RESULTS_JSON = DATA_PATH / "results.json"
CUTOFFS = [1, 5, 10]
MAX_K = max(CUTOFFS)

embeddings = EmbeddingModel()
reranker = RerankerModel()
llm = OpenAIClient()


# RAG & Save -------------------------------------------------------------------


def main() -> None:
    all_documents, all_chunks = DatasetDocument.load(DOCS_PATH)
    all_qa_pairs = DatasetQA.load(QA_PATH)

    results: list[dict] = []
    for doc in all_documents:
        doc_chunks = [chunk for chunk in all_chunks if chunk.doc_id == doc.id]
        doc_qa = [
            qa
            for qa in all_qa_pairs
            if qa.article_title == doc.title and qa.language == doc.language
        ]
        if len(doc_qa) == 0:
            continue

        # 1. Insert
        db = KnowledgeBase(name=doc.source, test=True)
        db.insert(
            docs=[doc],
            chunks=doc_chunks,
            embedding_model=embeddings,
        )

        # 2. RAG
        for qa in doc_qa:
            retrieved_chunks = db.search(
                query=qa.question,
                top_k=MAX_K,
                top_r=20,
                embedding_model=embeddings,
                reranker_model=reranker,
                threshold=0.6,
            )
            stream = generate_answer(
                query=qa.question,
                model=llm,
                history=None,
                chunks=None,
            )
            answer = "".join(stream)
            results.append({
                "user_input": qa.question,
                "retrieved_contexts": [c.chunk.text for c in retrieved_chunks],
                "retrieved_ids": [int(c.chunk.id) for c in retrieved_chunks],
                "ground_truth_ids": qa.chunks,
                "response": answer,
                "reference": qa.answer,
            })

    # Save Results
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation results to {RESULTS_JSON} :)")


if __name__ == "__main__":
    main()
