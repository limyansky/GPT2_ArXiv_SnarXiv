"""Tokenizer for GPT2 with Start, Stop, Pad"""

# external
from transformers import GPT2TokenizerFast

# internal
from .base_tokenizer import BaseTokenizer

class SnarTok(BaseTokenizer):
    """GPT2 Tokenizer for SnarXiv-tuned GPT2"""
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = None # Consider self.load here

    def load(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.config["tokenizer"]["name"],
                                                            eos_token=self.config["tokenizer"]
                                                                                 ["eos_token"],
                                                            bos_token=self.config["tokenizer"]
                                                                                 ["bos_token"],
                                                            pad_token=self.config["tokenizer"]
                                                                                 ["pad_token"])

    def tokenize(self, string):
        return self.tokenizer(string, padding=False)
