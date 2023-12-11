"""
Converts the .json datafile into strings for use in training.
"""
import re
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

    string = re.sub(r'[^a-zA-Z0-9 - : +]+', '', input_str)
    string = re.sub(r' +', ' ', string)

    return string
