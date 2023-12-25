"""Runs trains a model"""

from steps import (
    data_loader,
    process_data,
    tokenizer_loader,
    tokenization_step,
    model_trainer
    )

from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)

@pipeline
def SnarXiv_training_pipeline(
    ):
    """
    Model training pipeline.
    """

    ### Load Dataset Stage ###
    dataset = data_loader()

    ### Process Dataset Stage ###
    data = process_data(dataset)

    ### Load the tokenizer ###
    tokenizer = tokenizer_loader()

    ### Tokenize ###
    tokenized_data = tokenization_step(tokenizer, data)

    ### Training Stage ###
    model = model_trainer(tokenized_dataset=tokenized_data,
                          tokenizer=tokenizer
                         )

    return model
