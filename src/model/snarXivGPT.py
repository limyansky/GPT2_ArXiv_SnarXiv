"""The GPT2 LLM to generate paper titles"""

# external
from transformers import (TFGPT2LMHeadModel, AutoConfig)
import tensorflow as tf

# internal
from .base_model import BaseModel

class SnarGPT(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = self.build()

    def load_data(self):
        pass

    def build(self):
        """Loads GPT2 from huggingface transformers library"""

        # Generate configuration
        model_config = AutoConfig.from_pretrained(
            self.config["model"]["untuned"]["name"],
            bos_token_id = self.config["tokenizer"]["bos_token"],
            eos_token_id = self.config["tokenizer"]["eos_token"],
            pad_token_id = self.config["tokenizer"]["pad_token"],
            output_hidden_states = False)

        # Load model
        model = TFGPT2LMHeadModel.from_pretrained(
            self.config["model"]["untuned"]["name"],
            config=model_config)

        # Build model
        model(model.dummy_inputs)
        model.summary()

        if self.config["train"]["optimizer"]["type"] == "Adam":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate = self.config["train"]["optimizer"]["learning_rate"],
                epsilon = self.config["train"]["optimizer"]["epsilon"],
                clipnorm = self.config["train"]["optimizer"]["clipnorm"])
        else:
            exception_string = "Optimizer type {} is not currently supported."
            raise Exception(exception_string.format(self.config["train"]["optimizer"]["type"]))

        model.compile(optimizer)

        return model

    def save_weights(self):
        pass

    def load_weights(self):
        pass

    def train(self):
        pass