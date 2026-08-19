import timeit
from lib.models.llm import OpenAIClient


def benchmark(prompts, model, runs=5):
    """
    Ejecuta el microbenchmark y devuelve:
      - total_time: tiempo promedio total por ejecución (s)
      - throughput: tokens generados por segundo
    """
    # Warm-up
    for _ in range(3):
        model.generate(prompts[0])

    # Medir tiempo total
    def run_all():
        for p in prompts:
            model.generate(p)

    total_time = timeit.timeit(run_all, number=runs) / runs
    # Tokens aproximados: 1 token ≈ 4 caracteres
    total_tokens = sum(len(p) // 4 for p in prompts)
    throughput = total_tokens / total_time

    return total_time, throughput


def report(name, prompts, total, tput):
    """Imprime un resumen de los resultados."""
    print(
        f"{name:<6} | Prompts: {len(prompts):>2} | "
        f"Total: {total:>6.4f}s | "
        f"Throughput: {tput:>7.2f} tok/s"
    )


if __name__ == "__main__":
    model = OpenAIClient()

    # Definición de conjuntos de prueba
    prompt_sets = {
        "Short": [
            "Explain quantum computing in simple terms",
            "What is the capital of France?",
        ]
        * 2,  # ~ 34 tokens
        "Medium": [
            "Write a detailed comparison between Python and JavaScript for web development, considering factors like performance, ecosystem, and learning curve.",
            "Describe the process of photosynthesis in plants, including the light-dependent and light-independent reactions.",
        ]
        * 2,  # ~ 128 tokens
        "Long": [
            "Write a comprehensive essay about the history of artificial intelligence, covering its origins in the 1950s, key developments through the decades, current state-of-the-art techniques, and ethical considerations for future development. Include at least five major milestones and discuss the impact of AI on three different industries.",
            "In recent years, the rapid advancement of artificial intelligence and machine learning technologies has not only transformed the landscape of modern computing but has also raised critical ethical questions regarding privacy, fairness, and accountability, particularly as these systems are increasingly deployed in high-stakes domains such as healthcare.",
        ]
        * 2,  # ~ 342 tokens
    }

    print("== Benchmark Generación LLM ==\n")
    for name, prompts in prompt_sets.items():
        total, tput = benchmark(prompts, model)
        report(name, prompts, total, tput)
