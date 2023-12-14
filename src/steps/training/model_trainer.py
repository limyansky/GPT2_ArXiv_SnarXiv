"""Performs model training"""
from typing import Optional

from datasets import Dataset
from transformers import (
    PreTrainedTokenizerBase,
    DataCollatorForLanguageModeling,
    AutoConfig,
    TFGPT2LMHeadModel
    )
import tensorflow as tf
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

# An experiment tracker would go here.

@step
def model_trainer(
    tokenized_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    train_batch_size: Optional[int] = 32,
    ):
    """
    Configure and train model on training dataset.

    Args:
        tokenized_dataset: The tokenized dataset.
        tokenizer: The tokenizer.
        train_batch_size: Training batch size.

    Returns:
        The trained model.
    """

    ### Load the Model ###

    config = AutoConfig.from_pretrained(
        'gpt2',
        bos_token_id = tokenizer.bos_token_id,
        eos_token_id = tokenizer.eos_token_id,
        pad_token_id = tokenizer.pad_token_id,
        output_hidden_states = False
        )

    model = TFGPT2LMHeadModel.from_pretrained('gpt2', config=config)

    # Tell the model that we modified the tokenizer
    model.resize_token_embeddings(len(tokenizer))

    model(model.dummy_inputs) # Builds model

    logger.info(model.summary())

    # Compile model for training
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5,
                                         epsilon=1e-08, clipnorm=1.0)

    model.compile(optimizer)

    # Early Stopping callback
    cb_EarlyStopping = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        restore_best_weights = True,
        min_delta = 0.1,
        patience = 2)

    ### Prepare Data for GPT2 ###

    # Pull only the relevant columns from the dataset
    tokenized_dataset = tokenized_dataset.select_columns(["input_ids", "attention_mask"])

    # Create training and validation data
    # Can add 'seed' option for reproducibility
    train_valid_set = tokenized_dataset.train_test_split(test_size=0.1)

    # Used to prepare batches for training
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False,
                                                    return_tensors='tf')

    # Convert the huggingface datasets to something tensorflow can work with.
    tf_train = model.prepare_tf_dataset(
        train_valid_set["train"],
        collate_fn = data_collator,
        batch_size = train_batch_size)

    tf_valid = model.prepare_tf_dataset(
        train_valid_set["test"],
        collate_fn = data_collator,
        batch_size = train_batch_size)

    ### Final Configurations ###

    # Train in mixed-precision fload16 for speed
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    ### Train ###
    logger.info("Beginning training...")
    
    model.fit(tf_train, validation_data=tf_valid,
              epochs=100, callbacks=[cb_EarlyStopping])

    return model
