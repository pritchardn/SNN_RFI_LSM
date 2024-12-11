"""
This module contains the default configuration parameters for the different models.
"""

import copy
import os

DEFAULT_HERA_RATE = {
    "data_source": {
        "data_path": "./data",
        "limit": 1.0,
        "patch_size": 32,
        "stride": 32,
        "dataset": "HERA",
    },
    "dataset": {
        "batch_size": 36,
    },
    "model": {
        "type": "LSM",
        "num_inputs": 32,
        "num_hidden": 128,
        "num_outputs": 32,
    },
    "trainer": {
        "epochs": 100,
        "num_nodes": int(os.getenv("NNODES", 1)),
    },
    "encoder": {
        "method": "RATE_FULL",
        "exposure": 8,
        "tau": 1.0,
        "normalize": True,
    },
}


def get_default_params(
        dataset: str,
        model_type: str,
        model_size: int = 128,
        exposure_mode: str = None,
        delta_normalization: bool = False,
):
    if dataset == "HERA":
        if model_type == "LSM":
            params = copy.deepcopy(DEFAULT_HERA_RATE)
        else:
            raise ValueError(f"Unknown model type {model_type}")
    elif dataset == "LOFAR":
        raise ValueError(f"Unknown model type {model_type}")
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    params["model"]["type"] = model_type
    params["data_source"]["delta_normalization"] = delta_normalization
    return params
