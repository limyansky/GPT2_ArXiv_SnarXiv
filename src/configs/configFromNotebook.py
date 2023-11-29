"""Model config replicating ArXiv_PC_Training.ipynb"""

CFG = {
    "data": {
        "raw_path": "./data/raw/arxiv-metadata-oai-snapshot.json",  # Change to proper path
        "processed_path": "path/here"
    },
    "train": {
        "batch_size": 32,
        "buffer_size": 1,  # Need to set
        "epochs": 10,
        "precision": "mixed_float16",
        "optimizer": {
            "type": "Adam",
            "learning_rate": 3e-5,
            "epsilon": 1e-08,
            "clipnorm": 1.0
        },
        "metrics": ["accuracy"]
    },
    "model": {
        "untuned": {
            "name": "gpt2",
            "output_hidden_states": False
        },
        "tuned": {
        }  # What would I like to save here?
    },
    "tokenizer": {
        "name": "gpt2",
        "eos_token": "<|endoftext|>",
        "bos_token": "<|startoftext|>",
        "pad_token": "<pad>"
    }
}
