# 🧠 SmartQ Dataset Creator

![screenshot](paper/ui.png)

**SmartQ Dataset Creator** generates **synthetic question–answer datasets** from **SmartQ knowledge bases**, Wikipedia, and Large Language Models (LLMs).
It was developed to support the evaluation of **Retrieval-Augmented Generation (RAG)** systems, particularly [this RAG evaluator](https://github.com/humankernel/rag-revamped).

## 📚 Selected Wikipedia Topics

### 🧮 Mathematics

* [Prime Numbers](https://en.wikipedia.org/wiki/Prime_number)
* [Linear Algebra](https://en.wikipedia.org/wiki/Linear_algebra)
* [Calculus](https://en.wikipedia.org/wiki/Calculus)
* [Probability](https://en.wikipedia.org/wiki/Probability)

### 💻 Computer Science

* [Algorithm](https://en.wikipedia.org/wiki/Algorithm)
* [Data Structure](https://en.wikipedia.org/wiki/Data_structure)
* [Artificial Intelligence](https://en.wikipedia.org/wiki/Artificial_intelligence)
* [Computer Programming](https://en.wikipedia.org/wiki/Computer_programming)

### 🧬 Biology

* [Cell (biology)](https://en.wikipedia.org/wiki/Cell_%28biology%29)
* [Genetics](https://en.wikipedia.org/wiki/Genetics)
* [Evolution](https://en.wikipedia.org/wiki/Evolution)
* [Ecology](https://en.wikipedia.org/wiki/Ecology)

### ⚛️ Physics

* [Classical Mechanics](https://en.wikipedia.org/wiki/Classical_mechanics)
* [Electromagnetism](https://en.wikipedia.org/wiki/Electromagnetism)
* [Quantum Mechanics](https://en.wikipedia.org/wiki/Quantum_mechanics)
* [Thermodynamics](https://en.wikipedia.org/wiki/Thermodynamics)

### 🌍 General Topics

* [Batman](https://en.wikipedia.org/wiki/Batman)
* [Dachshund](https://en.wikipedia.org/wiki/Dachshund)
* [Conspiracy Theory](https://en.wikipedia.org/wiki/Conspiracy_theory)
* [Religion](https://en.wikipedia.org/wiki/Religion)

## 🧩 Question Types

Each dataset entry belongs to one of several **cognitive and reasoning categories**, enabling targeted evaluation of RAG models:

1. ✅ **Factual** – objective, verifiable facts.
2. 🔗 **Multi-Hop** – multi-step reasoning or combined facts.
3. 🧠 **Semantic** – interpretation and meaning of concepts.
4. ⚙️ **Logical Reasoning** – applying formal rules or laws.
5. 💡 **Creative Thinking** – open-ended or hypothetical reasoning.
6. 📏 **Problem-Solving** – applying formulas or methods to compute results.
7. ⚖️ **Ethical & Philosophical** – moral or conceptual reflection on science.

Each question type is designed to stress different aspects of retrieval and generation in RAG systems.

## ⚡ One-Click SmartQ Bulk Q/A Generation

The Gradio **Generate Q/A** tab can create a downloadable JSONL dataset from multiple SmartQ document IDs in one operation:

1. Enter document IDs separated by commas or new lines.
2. Set the requested total number of Q/A pairs.
3. The app generates `floor(total / document_count)` pairs per document.
4. For each document, the first and last 10 chunks are excluded, then non-adjacent chunks are selected randomly.
5. Each selected chunk produces exactly one factual Q/A pair, and all pairs are written to one UTF-8 JSONL download.

If integer division leaves a remainder, the UI reports how many requested pairs were not allocated. Documents that cannot supply the required number of non-adjacent interior chunks produce a clear validation error.

## 📊 Evaluation Metrics

Although SmartQ Dataset Creator only generates datasets, it is designed around **RAG evaluation metrics** (see [Key Metrics and Evaluation Methods for RAG](https://www.youtube.com/watch?v=cRz0BWkuwHg)).

### 🔍 Retrieval Metrics

| Metric                           | Measures                    | Description                                           |
| -------------------------------- | --------------------------- | ----------------------------------------------------- |
| **Precision**                    | Relevance of retrieved docs | Fraction of retrieved documents that are relevant     |
| **Recall**                       | Coverage of relevant docs   | Fraction of relevant documents that were retrieved    |
| **Hit Rate**                     | Top-result success          | % of queries retrieving ≥1 relevant doc in top-k      |
| **MRR (Mean Reciprocal Rank)**   | Top result position         | Measures how high the first relevant doc ranks        |
| **NDCG**                         | Ranking quality             | Evaluates both relevance and order of retrieved docs  |
| **MAP (Mean Average Precision)** | Overall retrieval accuracy  | Averages precision over all relevant docs and queries |

### ✍️ Generation Metrics

| Metric                 | Measures                              | Example                                              |
| ---------------------- | ------------------------------------- | ---------------------------------------------------- |
| **Faithfulness**       | Factual consistency with context      | “Einstein was born in Germany on March 14, 1879.”    |
| **Answer Relevance**   | How well the answer fits the question | Adds missing but relevant info like France → “Paris” |
| **Answer Correctness** | Alignment with ground truth           | Matches true reference answer accurately             |

## ⚙️ Example Use Case

This tool can be used to:

* Build **synthetic QA datasets** for RAG benchmark testing.
* Evaluate the **retrieval** and **generation** quality of LLM-based systems.
* Train or fine-tune **retrieval models** on domain-specific scientific content.

## 🧠 Related Projects

* 🔗 **RAG Evaluator:** [humankernel/rag-revamped](https://github.com/humankernel/rag-revamped)
* 🧾 **Undergraduate Thesis:** [humankernel/thesis](https://humankernel.github.io/thesis/main.pdf)