"""Model config replicating ArXiv_PC_Training.ipynb"""

CFG = {
    "data": {
        "raw_path": "path/here",
        "processed_path": "path/here"
    },
    "train": {
        "batch_size": 1,
        "buffer_size": 1,
        "epochs": 1,
        "optimizer": {
            "type": "optimizer name",
            "step": "step size"
        },
        "metrics": ["accuracy"]
    },
    "model": {
        "untuned": {
        },
        "tuned": {
        }

    }
}
