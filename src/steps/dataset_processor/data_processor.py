"""
Converts the .json datafile into strings for use in training.
"""
import re
import multiprocessing
from typing_extensions import Annotated
from datasets import Dataset
from zenml import step
from zenml.logger import get_logger


logger = get_logger(__name__)

def clean_txt(input_str: str) -> str:
    """Removes special characters.
    
    Special characters are defined using the regular expression [^a-zA-Z0-9 -:].
    Multiple spaces are redefined to single spaces.

    Args:
        input_str: The string to filter.

    Returns:
        A string which keeps only letters, numbers, (single) spaces, dashes, and colons.
    """

    string = re.sub(r'[^a-zA-Z0-9 \- : +]+', '', input_str)
    string = re.sub(r' +', ' ', string)

    return string

def training_string(data: Dataset) -> str:
    """ Generates a string in the proper format to train the model.

    Args:
        data: A Dataset containing information about each paper.

    Returns
        A string of the format: 
            "<|startoftext|> <|categories|> cat1 cat2 <|title|> Paper Title <|endoftext|>"
    """

    string_template = "<|startoftext|> <|categories|> {} <|title|> {} <|endoftext|>"

    cats = data["categories"]
    title = data["title"]

    title = clean_txt(title)

    return string_template.format(cats, title)

@step
def process_data(data: Dataset) -> Annotated[Dataset, "Processed Dataset"]:
    """ Cleans the paper titles and creates a properly formatted training string.

    Args:
        data: A Dataset containing information about each paper.

    Returns:
        A Dataset with titles overwritten by cleaned titles, and a 
        'training_string' element suitable for tokenization
    """

    def clean_txt_mapper(data: Dataset):
        data["title_clean"] = clean_txt(data["title"])
        return data

    def training_string_mapper(data: Dataset):
        data["training_string"] = training_string(data)
        return data

    logger.info("Cleaning paper titles...")
    data = data.map(clean_txt_mapper)

    logger.info("Creating training strings...")
    data = data.map(training_string_mapper)

    return data
