from typing import Literal

import gradio as gr
import pandas as pd

from .lib.llm import generate, get_client
from .lib.prompt import PROMPT
from .lib.types import QA, Article
from .lib.utils import create_json_file, format_context
from .lib.wikipedia import get_wikipedia_article

# --- Constants ---

SOURCES = ["Wikipedia"]
LANGUAGES = ["en", "es"]
TYPES_QUERIES = ["factual", "multihop"]
MAX_CHUNKS_PER_QA = 3
llm = get_client()

# --- Backend & Data Handling


def get_articles(
    source: Literal["Wikipedia"], title: str, langs: list[str]
) -> list[Article]:
    articles: list[Article] = []

    try:
        match source:
            case "Wikipedia":
                articles = get_wikipedia_article(title, langs)
            case _:
                raise ValueError(
                    f"Unsupported source: '{source}'. Only 'Wikipedia' is currently supported."
                )
    except ConnectionError as e:
        print(f"Network error while fetching from {source}: {str(e)}")
    except Exception as e:
        print(f"Unexpected error fetching articles: {str(e)}")

    return articles


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
        with gr.Row():
            source = gr.Dropdown(
                label="Sources",
                choices=SOURCES,
                value=SOURCES[0],
            )
            languages = gr.Dropdown(
                label="Language(s)",
                choices=LANGUAGES,
                value=LANGUAGES[0],
                multiselect=True,
            )
        title = gr.Textbox(
            label="Article Title",
            placeholder="Example: Artificial Intelligence",
            submit_btn=True,
        )
        title.submit(
            get_articles,
            inputs=[source, title, languages],
            outputs=[articles_state],
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
