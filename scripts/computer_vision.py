import logging
from typing import Annotated

import cv2 as cv
import typer
from dotenv import load_dotenv

from template_ml.loggers import get_logger
from template_ml.settings import Settings

# Initialize the Typer application
app = typer.Typer(
    add_completion=False,
    help="Computer Vision CLI",
)

# Set up logging
logger = get_logger(__name__)


@app.command()
def show(
    file_name: Annotated[
        str,
        typer.Option(
            "--file",
            "-f",
            help="Path to the video file to show",
        ),
    ] = "video.mp4",
    output_file: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Path to the output video file",
        ),
    ] = "output_video.mp4",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
):
    if verbose:
        logger.setLevel(logging.DEBUG)

    settings = Settings()
    logger.debug(f"Options: file_name={file_name}, output_file={output_file}, verbose={verbose}")
    logger.debug(f"Settings from .env: {settings.model_dump_json(indent=2)}")

    # Open the camera or file
    video_capture = cv.VideoCapture(file_name)
    if not video_capture.isOpened():
        logger.error("Cannot open camera")
        exit()

    video_writer = cv.VideoWriter(
        filename=output_file,
        fourcc=cv.VideoWriter_fourcc(*"mp4v"),
        fps=video_capture.get(cv.CAP_PROP_FPS),
        frameSize=(int(video_capture.get(cv.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))),
    )

    while True:
        # Read a frame from the video capture
        ret, frame = video_capture.read()
        if not ret:
            logger.info("End of video stream")
            break

        # Display the frame
        cv.imshow("Video", frame)

        # Write the frame to the video file
        video_writer.write(frame)

        # Exit on 'q' key press
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    video_capture.release()
    video_writer.release()
    cv.destroyAllWindows()

    logger.info("Released video capture and writer resources.")


if __name__ == "__main__":
    load_dotenv(
        override=True,
        verbose=True,
    )
    app()
