"""Loads the tokenizer"""

from transformers import GPT2TokenizerFast, PreTrainedTokenizerBase
from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def tokenizer_loader() -> Annotated[PreTrainedTokenizerBase, "tokenizer"]:
    """Initalizes the tokenizer.

    The GPT2 fast tokenizer is hard-coded.
    """

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2',
                                              eos_token='<|endoftext|>',
                                              bos_token='<|startoftext|>',
                                              pad_token='<pad>')

    return tokenizer
