"""
Functions to handle the loading of data.

At the moment, the path to the dataset is hardcoded.
"""

from typing_extensions import Annotated
from datasets import load_dataset, DatasetDict
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def data_loader() -> Annotated[DatasetDict, "dataset"]:
    """
    Reads a .json file containing the ArXiv dataset.

    Returns:
        The loaded dataset artifact.
    """

    dataset_path = "../../../data/raw/arxiv-metadata-oai-snapshot.json"

    logger.info("Loading dataset from {}".format(dataset_path))

    # I must specify a split, but no splitting is implimented
    data = load_dataset("json", data_files = dataset_path, split="train")

    logger.info(data)
    logger.info("Dataset Loaded Successfully")

    return data
