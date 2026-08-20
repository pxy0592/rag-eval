import random
import re
from typing import Literal

from .lib.proxy import remove_unsupported_socks_proxies

remove_unsupported_socks_proxies()

import gradio as gr
import pandas as pd

from .lib.llm import generate, get_client
from .lib.prompt import PROMPT
from .lib.smartq import (
    SmartQAPIError,
    SmartQChunkPage,
    SmartQClient,
    SmartQKnowledgePage,
)
from .lib.types import QA, QAFormat, Article, Chunk
from .lib.utils import (
    create_json_file,
    create_jsonl_file,
    format_context,
    parse_qa_output,
)
from .lib.wikipedia import get_wikipedia_article
from .settings import settings

# --- Constants ---

SOURCES = ["SmartQ", "Wikipedia"]
LANGUAGES = ["en", "es", "cn"]
TYPES_QUERIES = ["factual", "multihop"]
MAX_CHUNKS_PER_QA = 3
SMARTQ_PAGE_SIZE = 20
llm = get_client()

# --- Backend & Data Handling


def get_articles(source: str, title: str, langs: list[str]) -> list[Article]:
    """Fetch Wikipedia articles; SmartQ uses its knowledge-base controls instead."""

    articles: list[Article] = []
    try:
        if source != "Wikipedia":
            raise ValueError(
                f"Unsupported source: '{source}'. Use the SmartQ knowledge-base controls"
            )
        articles = get_wikipedia_article(title, langs)
    except ConnectionError as error:
        print(f"Network error while fetching from {source}: {error}")
    except Exception as error:
        print(f"Unexpected error fetching articles: {error}")
    return articles


def get_smartq_knowledge_page(
    knowledge_base_id: str | None, page: int
) -> SmartQKnowledgePage:
    """Fetch one server-backed, 20-item page of SmartQ knowledge titles."""

    client = SmartQClient(settings.SMARTQ_API_URL, settings.SMARTQ_API_KEY)
    return client.list_knowledge_page(knowledge_base_id, page, SMARTQ_PAGE_SIZE)


def get_smartq_chunk_page(
    knowledge_id: str | None, page: int
) -> SmartQChunkPage:
    """Fetch one server-backed, 20-item page of SmartQ text chunks."""

    client = SmartQClient(settings.SMARTQ_API_URL, settings.SMARTQ_API_KEY)
    return client.get_chunk_page(knowledge_id, page, SMARTQ_PAGE_SIZE)


def _page_status(label: str, page: SmartQKnowledgePage | SmartQChunkPage) -> str:
    if page.total == 0:
        return f"No {label.lower()} found."
    first = (page.page - 1) * page.page_size + 1
    last = min(first + len(page.items) - 1, page.total)
    total_pages = max(1, (page.total + page.page_size - 1) // page.page_size)
    return f"{label}: {first}-{last} of {page.total} (page {page.page}/{total_pages})"


def _knowledge_page_updates(knowledge_base_id: str | None, page: int):
    try:
        knowledge_page = get_smartq_knowledge_page(knowledge_base_id, page)
    except SmartQAPIError as error:
        raise gr.Error(str(error)) from error

    choices = [(knowledge.title, knowledge.id) for knowledge in knowledge_page.items]
    return (
        gr.update(choices=choices, value=None),
        knowledge_base_id,
        knowledge_page.page,
        _page_status("Knowledges", knowledge_page),
        gr.update(interactive=knowledge_page.has_previous),
        gr.update(interactive=knowledge_page.has_next),
    )


def load_smartq_knowledge_page(knowledge_base_id: str | None):
    """Load the first server-backed page after a knowledge base is entered."""

    return _knowledge_page_updates(knowledge_base_id, 1)


def previous_smartq_knowledge_page(knowledge_base_id: str | None, page: int):
    return _knowledge_page_updates(knowledge_base_id, max(1, page - 1))


def next_smartq_knowledge_page(knowledge_base_id: str | None, page: int):
    return _knowledge_page_updates(knowledge_base_id, page + 1)


def _chunk_rows(chunk_page: SmartQChunkPage) -> list[list[str | int]]:
    return [[chunk.index + 1, chunk.content] for chunk in chunk_page.items]


def _chunk_page_updates(knowledge_id: str | None, page: int):
    try:
        chunk_page = get_smartq_chunk_page(knowledge_id, page)
    except SmartQAPIError as error:
        raise gr.Error(str(error)) from error

    return (
        _chunk_rows(chunk_page),
        chunk_page.page,
        _page_status("Chunks", chunk_page),
        gr.update(interactive=chunk_page.has_previous),
        gr.update(interactive=chunk_page.has_next),
    )


def previous_smartq_chunk_page(knowledge_id: str | None, page: int):
    return _chunk_page_updates(knowledge_id, max(1, page - 1))


def next_smartq_chunk_page(knowledge_id: str | None, page: int):
    return _chunk_page_updates(knowledge_id, page + 1)


def fetch_smartq_article(knowledge_id: str | None):
    """Fetch all chunks for Q/A and the first server-backed display page."""

    try:
        client = SmartQClient(settings.SMARTQ_API_URL, settings.SMARTQ_API_KEY)
        article = client.get_article(knowledge_id)
    except SmartQAPIError as error:
        raise gr.Error(str(error)) from error

    chunk_updates = _chunk_page_updates(knowledge_id, 1)
    return ([article], knowledge_id, *chunk_updates)


def update_source_controls(source: str):
    """Show the controls appropriate to the selected article source."""

    is_smartq = source == "SmartQ"
    return gr.update(visible=not is_smartq), gr.update(visible=is_smartq)


def qa_pair_from_chunks(chunks: list[Chunk]) -> QAFormat | None:
    """Return the first Q/A pair already present in a selected chunk."""

    for chunk in chunks:
        parsed_output = parse_qa_output(chunk.content)
        if parsed_output:
            return QAFormat.model_validate(parsed_output)
    return None


_ANSWER_SENTENCE_MARKERS = re.compile(
    r"(?:是|为|有|共有|包括|位于|达到|属于|发生|表示|指|由|在)"
    r"|(?:\b(?:is|are|was|were|has|have|contains|includes|equals|"
    r"es|son|fue|eran|tiene|incluye)\b)",
    re.IGNORECASE,
)


def answer_needs_expansion(answer: str) -> bool:
    """Identify short factual fragments that should become standalone sentences."""
    compact = re.sub(r"[^\w\u4e00-\u9fff]+", "", answer, flags=re.UNICODE)
    if not compact:
        return True
    if _ANSWER_SENTENCE_MARKERS.search(answer):
        return False
    return len(compact) <= 24


def expand_factual_answer(
    qa_pair: QAFormat, article: Article, context: str
) -> QAFormat:
    """Use one repair pass when the generated answer is only a short fragment."""
    if not answer_needs_expansion(qa_pair.answer):
        return qa_pair
    prompt = PROMPT["factual_answer_rewrite"].format(
        article_title=article.title,
        question=qa_pair.question,
        answer=qa_pair.answer,
        context=context,
    )
    rewritten = generate(prompt=prompt, llm=llm)
    return QAFormat(question=qa_pair.question, answer=rewritten.answer)


def create_synthetic_qa_pair(
    type_q: Literal["factual", "multihop"],
    article: Article,
    chunks_idx: list[int],
) -> QAFormat:
    """Generate or refine one structured Q/A pair from selected chunks."""
    if not chunks_idx:
        raise ValueError("Select at least one context chunk")
    chunks = [article.chunks[chunk_idx] for chunk_idx in chunks_idx]
    qa_pair = qa_pair_from_chunks(chunks)
    context = format_context(chunks)
    if qa_pair is None:
        match type_q:
            case "factual":
                prompt = PROMPT["factual_qa_pair"].format(
                    article_title=article.title,
                    context=context,
                )
            case _:
                raise ValueError(f"Unsupported question type: {type_q}")
        qa_pair = generate(prompt=prompt, llm=llm)
    if type_q == "factual":
        qa_pair = expand_factual_answer(qa_pair, article, context)
    return qa_pair


def generate_syntetic_qa_pair(
    type_q: Literal["factual", "multihop"],
    article: Article,
    chunks_idx: list[int],
):
    """Gradio callback for manually generating one Q/A pair."""
    try:
        qa_pair = create_synthetic_qa_pair(type_q, article, chunks_idx)
        return (
            gr.update(
                value=qa_pair.question,
                visible=True,
                interactive=True,
            ),
            gr.update(
                value=qa_pair.answer,
                visible=True,
                interactive=True,
            ),
        )
    except Exception as error:
        raise gr.Error(f"Error to generate {error}") from error


def parse_smartq_document_ids(raw_ids: str) -> list[str]:
    """Parse comma/newline-separated SmartQ document IDs without duplicates."""
    document_ids = [
        value.strip()
        for value in re.split(r"[,\n\r]+", raw_ids or "")
        if value.strip()
    ]
    return list(dict.fromkeys(document_ids))


def select_random_non_adjacent_chunks(
    total_chunks: int,
    count: int,
    rng: random.Random | None = None,
) -> list[int]:
    """Choose random non-adjacent indexes after excluding ten at each end."""
    if count < 1:
        raise ValueError("Q/A count per document must be at least 1")
    candidate_count = max(0, total_chunks - 20)
    max_non_adjacent = (candidate_count + 1) // 2
    if count > max_non_adjacent:
        raise ValueError(
            f"Document has {total_chunks} chunks; after excluding the "
            "first and "
            f"last 10 chunks, at most {max_non_adjacent} non-adjacent chunks "
            f"can be selected, but {count} were requested"
        )

    generator = rng or random.Random()
    compressed_positions = sorted(
        generator.sample(range(candidate_count - count + 1), count)
    )
    return [
        10 + compressed_position + offset
        for offset, compressed_position in enumerate(compressed_positions)
    ]


def generate_bulk_smartq_qa(
    raw_document_ids: str,
    total_qa_count: int | float,
    type_q: Literal["factual", "multihop"] = "factual",
    *,
    client: SmartQClient | None = None,
    rng: random.Random | None = None,
) -> tuple[list[QA], str]:
    """Generate floor(total/documents) one-chunk Q/A pairs per document."""
    document_ids = parse_smartq_document_ids(raw_document_ids)
    if not document_ids:
        raise ValueError("Enter at least one SmartQ document ID")
    if (
        isinstance(total_qa_count, bool)
        or int(total_qa_count) != total_qa_count
    ):
        raise ValueError("Total Q/A count must be an integer")
    requested_total = int(total_qa_count)
    if requested_total < 1:
        raise ValueError("Total Q/A count must be at least 1")

    per_document = requested_total // len(document_ids)
    if per_document < 1:
        raise ValueError(
            "Total Q/A count must be at least the number of document IDs"
        )

    smartq_client = client or SmartQClient(
        settings.SMARTQ_API_URL, settings.SMARTQ_API_KEY
    )
    generator = rng or random.Random()
    qa_pairs: list[QA] = []
    for document_id in document_ids:
        article = smartq_client.get_article(document_id)
        selected_indices = select_random_non_adjacent_chunks(
            len(article.chunks), per_document, generator
        )
        for chunk_index in selected_indices:
            qa_pair = create_synthetic_qa_pair(type_q, article, [chunk_index])
            qa_pairs.append(
                QA(
                    type=type_q,
                    language=article.language,
                    article_title=article.title,
                    chunks=[chunk_index],
                    question=qa_pair.question,
                    answer=qa_pair.answer,
                )
            )

    generated_total = len(qa_pairs)
    remainder = requested_total - generated_total
    status = (
        f"Generated {generated_total} Q/A pairs from {len(document_ids)} "
        f"documents ({per_document} per document)."
    )
    if remainder:
        status += (
            f" Requested total {requested_total} leaves a remainder of "
            f"{remainder} after integer division."
        )
    return qa_pairs, status


def generate_bulk_smartq_qa_file(
    raw_document_ids: str,
    total_qa_count: int | float,
    type_q: Literal["factual", "multihop"],
):
    """Gradio callback that generates bulk Q/A pairs and a JSONL download."""
    try:
        qa_pairs, status = generate_bulk_smartq_qa(
            raw_document_ids, total_qa_count, type_q
        )
        file_path = create_jsonl_file(
            [qa.to_json() for qa in qa_pairs], prefix="smartq_qa_bulk_"
        )
        return qa_pairs, file_path, status
    except (SmartQAPIError, ValueError) as error:
        raise gr.Error(str(error)) from error


def dataset_source_name(article_data: list[dict], qa_data: list[dict] | None = None) -> str:
    """Infer a stable download filename source from the fetched article data."""

    qa_titles = {str(qa.get("article_title", "")) for qa in qa_data or []}
    matching_articles = [
        article
        for article in article_data
        if not qa_titles or str(article.get("title", "")) in qa_titles
    ]
    source_names = {
        "smartq"
        if "/api/v1/knowledge/" in str(article.get("source", ""))
        else "wikipedia"
        if "wikipedia.org" in str(article.get("source", ""))
        else "dataset"
        for article in matching_articles
    }
    return source_names.pop() if len(source_names) == 1 else "dataset"


def add_to_qa_dataset(
    type: str,
    language: str,
    article_title: str,
    chunks: list[int],
    question: str,
    answer: str,
    qa_data: list[QA],
) -> list[QA]:
    qa = QA(
        type=type,
        language=language,
        article_title=article_title,
        chunks=chunks,
        question=question,
        answer=answer,
    )
    qa_data.append(qa)
    return qa_data


# --- UI Builder Functions ---


def build_article_tab(articles_state: gr.State) -> None:
    with gr.Tab("(1) Get Article"):
        source = gr.Dropdown(
            label="Source",
            choices=SOURCES,
            value=SOURCES[0],
        )

        with gr.Group(visible=False) as wikipedia_controls:
            with gr.Row():
                languages = gr.Dropdown(
                    label="Language(s)",
                    choices=LANGUAGES,
                    value=LANGUAGES[0],
                    multiselect=True,
                )
                title = gr.Textbox(
                    label="Wikipedia article title",
                    placeholder="Example: Artificial Intelligence",
                    submit_btn=True,
                )
            title.submit(
                get_articles,
                inputs=[source, title, languages],
                outputs=[articles_state],
            )

        with gr.Group(visible=True) as smartq_controls:
            gr.Markdown(
                "SmartQ knowledge and chunk lists load 20 items per page from the "
                "server. Chinese titles are displayed unchanged."
            )
            knowledge_base_id = gr.Textbox(label="SmartQ knowledge base ID")
            knowledge_base_state = gr.State("")
            knowledge_page_state = gr.State(1)
            list_knowledges_button = gr.Button("List SmartQ knowledge")
            knowledge_id = gr.Radio(
                label="SmartQ knowledges",
                choices=[],
                interactive=True,
            )
            knowledge_page_status = gr.Markdown("Enter a knowledge base ID to begin.")
            with gr.Row():
                previous_knowledge_button = gr.Button("Previous", interactive=False)
                next_knowledge_button = gr.Button("Next", interactive=False)

            selected_knowledge_state = gr.State("")
            chunk_page_state = gr.State(1)
            fetch_smartq_button = gr.Button("Fetch selected SmartQ article")
            smartq_chunks = gr.Dataframe(
                headers=["Chunk", "Content"],
                datatype=["number", "str"],
                label="SmartQ chunks",
                interactive=False,
                wrap=True,
            )
            chunk_page_status = gr.Markdown("Select and fetch a knowledge entry to list chunks.")
            with gr.Row():
                previous_chunk_button = gr.Button("Previous", interactive=False)
                next_chunk_button = gr.Button("Next", interactive=False)

            list_knowledges_button.click(
                load_smartq_knowledge_page,
                inputs=[knowledge_base_id],
                outputs=[
                    knowledge_id,
                    knowledge_base_state,
                    knowledge_page_state,
                    knowledge_page_status,
                    previous_knowledge_button,
                    next_knowledge_button,
                ],
            )
            previous_knowledge_button.click(
                previous_smartq_knowledge_page,
                inputs=[knowledge_base_state, knowledge_page_state],
                outputs=[
                    knowledge_id,
                    knowledge_base_state,
                    knowledge_page_state,
                    knowledge_page_status,
                    previous_knowledge_button,
                    next_knowledge_button,
                ],
            )
            next_knowledge_button.click(
                next_smartq_knowledge_page,
                inputs=[knowledge_base_state, knowledge_page_state],
                outputs=[
                    knowledge_id,
                    knowledge_base_state,
                    knowledge_page_state,
                    knowledge_page_status,
                    previous_knowledge_button,
                    next_knowledge_button,
                ],
            )
            fetch_smartq_button.click(
                fetch_smartq_article,
                inputs=[knowledge_id],
                outputs=[
                    articles_state,
                    selected_knowledge_state,
                    smartq_chunks,
                    chunk_page_state,
                    chunk_page_status,
                    previous_chunk_button,
                    next_chunk_button,
                ],
            )
            previous_chunk_button.click(
                previous_smartq_chunk_page,
                inputs=[selected_knowledge_state, chunk_page_state],
                outputs=[
                    smartq_chunks,
                    chunk_page_state,
                    chunk_page_status,
                    previous_chunk_button,
                    next_chunk_button,
                ],
            )
            next_chunk_button.click(
                next_smartq_chunk_page,
                inputs=[selected_knowledge_state, chunk_page_state],
                outputs=[
                    smartq_chunks,
                    chunk_page_state,
                    chunk_page_status,
                    previous_chunk_button,
                    next_chunk_button,
                ],
            )

        source.change(
            update_source_controls,
            inputs=[source],
            outputs=[wikipedia_controls, smartq_controls],
        )

        @gr.render([articles_state])
        def display_articles(fetched_articles: list[Article]):
            if not fetched_articles:
                gr.Markdown("No articles fetched yet.")
                return

            gr.Markdown(f"Fetched {len(fetched_articles)} article(s):")
            for article in fetched_articles:
                with gr.Accordion(
                    f"({article.language}) - {article.title}", open=False
                ):
                    gr.Markdown("**Summary:**")
                    gr.Markdown(article.summary or "No summary available.")

                    if article.language != "cn":
                        gr.Markdown("**Chunks:**")
                        if article.chunks:
                            df = pd.DataFrame(article.chunks)
                            gr.DataFrame(df, wrap=True)
                        else:
                            gr.Markdown("No chunks found for this article.")


def build_qa_tab(articles_state: gr.State, qa_data_state: gr.State) -> None:
    with gr.Tab("(2) Generate Q/A"):
        with gr.Row():
            type_q = gr.Dropdown(
                label="Question Type",
                choices=TYPES_QUERIES,
                value=TYPES_QUERIES[0],
                interactive=True,
                scale=3,
            )
            qa_counter = gr.Number(
                1, label="Q/A count", minimum=1, scale=1, interactive=True
            )

        with gr.Accordion(
            "One-click SmartQ bulk generation", open=False
        ):
            gr.Markdown(
                "Enter multiple SmartQ document IDs separated by commas or new "
                "lines. The requested total is divided evenly using integer "
                "division. Each Q/A uses one randomly selected, non-adjacent "
                "chunk after excluding the first and last 10 chunks."
            )
            bulk_document_ids = gr.Textbox(
                label="SmartQ document IDs",
                placeholder="document-id-1\ndocument-id-2",
                lines=4,
            )
            bulk_total = gr.Number(
                value=10,
                label="Total Q/A pairs",
                minimum=1,
                precision=0,
            )
            bulk_generate_button = gr.Button(
                "Generate Q/A JSONL", variant="primary"
            )
            bulk_status = gr.Markdown()
            bulk_file = gr.File(
                label="Download generated Q/A JSONL", file_count="single"
            )
            bulk_generate_button.click(
                generate_bulk_smartq_qa_file,
                inputs=[bulk_document_ids, bulk_total, type_q],
                outputs=[qa_data_state, bulk_file, bulk_status],
            )

        gr.Markdown("### Generate Questions per Article")

        @gr.render([articles_state, qa_counter])
        def display_select_chunks(
            fetched_articles: list[Article], qa_counter: int
        ):
            if not fetched_articles:
                gr.Markdown("Fetch an article in Tab (1) first.")
                return

            # Show accordion with list of QA for each lang
            for article in fetched_articles:
                with gr.Accordion(
                    f"({article.language}) - {article.title}", open=True
                ):
                    for i in range(qa_counter):
                        chunks = gr.Dropdown(
                            label="Select Context Chunks",
                            choices=[
                                (chunk["heading"], id)
                                for id, chunk in enumerate(article.chunks)
                            ],
                            multiselect=True,
                            max_choices=3,
                            container=False,
                        )
                        question = gr.Textbox(label="Question", visible=False)
                        answer = gr.TextArea(
                            label="Answer", visible=False, lines=5
                        )
                        generate_qa_button = gr.Button("Generate")
                        generate_qa_button.click(
                            lambda type_q,
                            chunks_idx,
                            art=article: generate_syntetic_qa_pair(
                                type_q=type_q,
                                article=art,
                                chunks_idx=chunks_idx,
                            ),
                            inputs=[type_q, chunks],
                            outputs=[question, answer],
                        ).then(
                            lambda type,
                            chunks,
                            question,
                            answer,
                            qa_data_state: add_to_qa_dataset(
                                type=type,
                                language=article.language,
                                article_title=article.title,
                                chunks=chunks,
                                question=question,
                                answer=answer,
                                qa_data=qa_data_state,
                            ),
                            inputs=[
                                type_q,
                                chunks,
                                question,
                                answer,
                                qa_data_state,
                            ],
                            outputs=[qa_data_state],
                        )


def build_save_tab(articles_state: gr.State, qa_data_state: gr.State) -> None:
    with gr.Tab("(3) Save Dataset"):
        gr.Markdown("### Review and Download Data")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("**Fetched Articles (JSON)**")
                article_json = gr.JSON(label="Articles Data")
                article_file = gr.File(
                    label="Download Article JSON File", file_count="single"
                )
                download_articles_button = gr.DownloadButton(
                    "Download Articles JSON", variant="primary"
                )
            with gr.Column(scale=3):
                gr.Markdown("**Generated Q/A Pairs (JSON)**")
                qa_json = gr.JSON(label="Q/A Data")
                qa_file = gr.File(
                    label="Download Q/A JSON File", file_count="single"
                )
                download_qa_button = gr.DownloadButton(
                    "Download Q/A JSON", variant="primary"
                )

        articles_state.change(
            lambda articles: [a.to_json() for a in articles],
            inputs=[articles_state],
            outputs=[article_json],
        )

        qa_data_state.change(
            lambda qa_pairs: [qa.to_json() for qa in qa_pairs],
            inputs=[qa_data_state],
            outputs=[qa_json],
        )

        def handle_article_download_click(article_data: list[dict]) -> str:
            if not article_data:
                raise gr.Error("No article data to download.")
            source_name = dataset_source_name(article_data)
            return create_json_file(article_data, prefix=f"{source_name}_articles_")

        def handle_qa_download_click(
            qa_data: list[dict], article_data: list[dict]
        ) -> str:
            if not qa_data:
                raise gr.Error("No Q/A data to download.")
            source_name = dataset_source_name(article_data, qa_data)
            return create_json_file(qa_data, prefix=f"{source_name}_qa_")

        download_articles_button.click(
            fn=handle_article_download_click,
            inputs=[article_json],
            outputs=[article_file],
        )
        download_qa_button.click(
            fn=handle_qa_download_click,
            inputs=[qa_json, article_json],
            outputs=[qa_file],
        )


# --- Main Application Launch ---


def launch() -> None:
    with gr.Blocks() as demo:
        articles = gr.State([])
        qa_data = gr.State([])

        gr.Markdown("# SmartQ: Dataset Generator")
        gr.Markdown(
            "Generate Question/Answer pairs from SmartQ knowledge or Wikipedia articles. "
            "**Workflow:** (1) Fetch Article -> (2) Generate Q/A -> (3) Save Dataset."
        )

        # --- Build UI Tabs ---
        build_article_tab(articles)
        build_qa_tab(articles, qa_data)
        build_save_tab(articles, qa_data)

    demo.launch()


if __name__ == "__main__":
    launch()
