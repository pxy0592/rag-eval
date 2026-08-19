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
from .lib.types import QA, Article
from .lib.utils import create_json_file, format_context
from .lib.wikipedia import get_wikipedia_article
from .settings import settings

# --- Constants ---

SOURCES = ["Wikipedia", "SmartQ"]
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


def generate_syntetic_qa_pair(
    type_q: Literal["factual", "multihop"],
    article: Article,
    chunks_idx: list[int],
):
    chunks = [article.chunks[chunk_idx] for chunk_idx in chunks_idx]
    context = format_context(chunks)
    match type_q:
        case "factual":
            prompt = PROMPT["factual_qa_pair"].format(context=context)

    try:
        qa_pair = generate(prompt=prompt, llm=llm)
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
    except Exception as e:
        raise gr.Error(f"Error to generate {e}")


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

        with gr.Group(visible=True) as wikipedia_controls:
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

        with gr.Group(visible=False) as smartq_controls:
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
            return create_json_file(article_data, prefix="wiki_articles_")

        def handle_qa_download_click(qa_data: list[dict]) -> str:
            if not qa_data:
                raise gr.Error("No Q/A data to download.")
            return create_json_file(qa_data, prefix="wiki_qa_")

        download_articles_button.click(
            fn=handle_article_download_click,
            inputs=[article_json],
            outputs=[article_file],
        )
        download_qa_button.click(
            fn=handle_qa_download_click,
            inputs=[qa_json],
            outputs=[qa_file],
        )


# --- Main Application Launch ---


def launch() -> None:
    with gr.Blocks() as demo:
        articles = gr.State([])
        qa_data = gr.State([])

        gr.Markdown("# WikiQA: Dataset Generator")
        gr.Markdown(
            "Generate Question/Answer pairs from Wikipedia articles."
            "**Workflow:** (1) Fetch Article -> (2) Generate Q/A -> (3) Save Dataset."
        )

        # --- Build UI Tabs ---
        build_article_tab(articles)
        build_qa_tab(articles, qa_data)
        build_save_tab(articles, qa_data)

    demo.launch()


if __name__ == "__main__":
    launch()
