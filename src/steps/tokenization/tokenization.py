"""Functions for performing the tokenization of text."""

from transformers import PreTrainedTokenizerBase
from datasets import DatasetDict
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def tokenization_step(
    tokenizer: PreTrainedTokenizerBase,
    dataset: DatasetDict)
    -> Annotated[DatasetDict, "tokenized_data"]:
    """
    Tokenization step. 
    """

