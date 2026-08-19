import timeit

from lib.models.embedding import EmbeddingModel


def benchmark_embedding_model(
    model: EmbeddingModel,
    sentences: list[str],
    batch_size: int = 32,
    num_runs: int = 10,
):
    """Benchmark the EmbeddingModel with various configurations."""

    def run_benchmark(name: str, **encode_args):
        def wrapper():
            return model.encode(sentences, **encode_args)

        time_taken = timeit.timeit(wrapper, number=num_runs) / num_runs
        print(f"{name}: {time_taken:.4f}s (avg over {num_runs} runs)")

    print(f"\nBenchmarking with {len(sentences)} sentences, batch_size={batch_size}")

    # Warm-up
    for _ in range(3):
        model.encode(
            sentences,
            return_dense=True,
            return_sparse=True,
            return_colbert=True,
        )

    # Benchmark different configurations
    run_benchmark(
        "Dense only",
        return_dense=True,
        return_sparse=False,
        return_colbert=False,
        batch_size=batch_size,
    )
    run_benchmark(
        "Sparse only",
        return_dense=False,
        return_sparse=True,
        return_colbert=False,
        batch_size=batch_size,
    )
    run_benchmark(
        "Colbert only",
        return_dense=False,
        return_sparse=False,
        return_colbert=True,
        batch_size=batch_size,
    )
    run_benchmark(
        "Dense + Sparse",
        return_dense=True,
        return_sparse=True,
        return_colbert=False,
        batch_size=batch_size,
    )
    run_benchmark(
        "All (Dense + Sparse + Colbert)",
        return_dense=True,
        return_sparse=True,
        return_colbert=True,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    model = EmbeddingModel()

    sentences_short = [
        "BGE M3 is an embedding model supporting dense retrieval.",
        "BM25 is a bag-of-words retrieval function.",
    ] # 2 sentences ~ 24 tokens
    sentences_long = sentences_short * 50  # 100 sentences ~ 1249 tokens
    sentences_long_long = sentences_long * 10  # 1000 sentences ~ 12499 tokens

    print("=== Benchmark (short input) ===\n")
    benchmark_embedding_model(model, sentences_short, batch_size=32)

    print("=== Benchmark (longer input) ===\n")
    benchmark_embedding_model(model, sentences_long, batch_size=32)

    print("=== Benchmark (very long input) ===\n")
    benchmark_embedding_model(model, sentences_long_long, batch_size=32)
