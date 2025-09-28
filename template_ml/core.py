from logging import DEBUG

from template_ml.loggers import get_logger

logger = get_logger(__name__)


def hello_world(verbose: bool = False):
    if verbose:
        logger.setLevel(DEBUG)
    logger.debug("Hello World")


if __name__ == "__main__":
    hello_world(verbose=True)
