"""
This module contains the default configuration parameters for the different models.
"""

import copy
import os

DEFAULT_HERA_LATENCY = {
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
        "type": "FC_LATENCY_ROCKPOOL",
        "num_inputs": 32,
        "num_hidden": 128,
        "num_outputs": 32,
        "num_layers": 6,
        "beta": 0.245507490258551,
        "tau_mem": 1.0,
        "tau_syn": 1.0,
        "learning_rate": 1e-3,
    },
    "trainer": {
        "epochs": 100,
        "num_nodes": int(os.getenv("NNODES", 1)),
    },
    "encoder": {
        "method": "LATENCY_FULL",
        "exposure": 64,
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
        if model_type == "FC_LATENCY":
            params = copy.deepcopy(DEFAULT_HERA_LATENCY)
        else:
            raise ValueError(f"Unknown model type {model_type}")
    elif dataset == "LOFAR":
        raise ValueError(f"Unknown model type {model_type}")
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    params["model"]["type"] = model_type
    params["data_source"]["delta_normalization"] = delta_normalization
    return params
