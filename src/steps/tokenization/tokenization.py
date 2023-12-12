"""Functions for performing the tokenization of text."""

from typing_extensions import Annotated
from transformers import PreTrainedTokenizerBase
from datasets import Dataset
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def tokenization_step(
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset
    ) -> Annotated[Dataset, "tokenized_data"]:
    """
    Tokenization step. 
    """

    def tokenizer_mapper(data:Dataset):
        tokenized_data = tokenizer(data["training_string"], padding=False)
        return tokenized_data

    dataset = dataset.map(tokenizer_mapper)

    return dataset
