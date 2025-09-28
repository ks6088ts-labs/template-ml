import logging
from typing import Annotated

import torch
import typer
from dotenv import load_dotenv
from transformers import pipeline

from template_ml.loggers import get_logger

app = typer.Typer(
    add_completion=False,
    help="Transformers Playground CLI",
)

logger = get_logger(__name__)
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B"


def set_verbose_logging(
    verbose: bool,
):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)


@app.command(help="ref. https://github.com/huggingface/transformers/blob/v4.56.2/README.md#quickstart")
def quickstart_pipeline(
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Model to use",
        ),
    ] = DEFAULT_MODEL,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = True,
):
    set_verbose_logging(verbose)
    pipeline_obj = pipeline(
        task="text-generation",
        model=model,
    )
    print(pipeline_obj("the secret to baking a really good cake is "))


@app.command(help="ref. https://github.com/huggingface/transformers/blob/v4.56.2/README.md#quickstart")
def quickstart_chat(
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Model to use",
        ),
    ] = DEFAULT_MODEL,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = True,
):
    set_verbose_logging(verbose)

    logger.info(
        f"You can also chat with a model directly from the command line using `uv run transformers chat {model}`"  # noqa: E501
    )

    chat = [
        {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
        {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"},
    ]

    pipeline_obj = pipeline(
        task="text-generation",
        model=model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    response = pipeline_obj(
        chat,
        max_new_tokens=512,
    )
    print(response[0]["generated_text"][-1]["content"])


if __name__ == "__main__":
    assert load_dotenv(
        override=True,
        verbose=True,
    ), "Failed to load environment variables"
    app()
