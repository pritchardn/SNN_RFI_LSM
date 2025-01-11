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
        "num_hidden": 8192,
        "num_outputs": 32,
        "probability_in": 0.2,
    },
    "trainer": {
        "epochs": 100,
        "num_nodes": int(os.getenv("NNODES", 1)),
    },
    "encoder": {
        "method": "RATE_FULL",
        "exposure": 16,
        "tau": 1.0,
        "normalize": True,
    },
}

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
        "type": "LSM",
        "num_inputs": 32,
        "num_hidden": 4092,
        "num_outputs": 32,
        "probability_in": 0.3,
    },
    "trainer": {
        "epochs": 100,
        "num_nodes": int(os.getenv("NNODES", 1)),
    },
    "encoder": {
        "method": "LATENCY_FULL",
        "exposure": 4,
        "tau": 1.0,
        "normalize": True,
    },
}

DEFAULT_HERA_DIRECT = {
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
        "num_hidden": 4096,
        "num_outputs": 32,
        "probability_in": 0.654,
    },
    "trainer": {
        "epochs": 100,
        "num_nodes": int(os.getenv("NNODES", 1)),
    },
    "encoder": {
        "method": "DIRECT",
        "exposure": 1,
        "tau": 1.0,
        "normalize": True,
    },
}

DEFAULT_HERA_RATE_REL = {
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
        "num_hidden": 8192,
        "num_outputs": 64,
        "probability_in": 0.2,
    },
    "trainer": {
        "epochs": 100,
        "num_nodes": int(os.getenv("NNODES", 1)),
    },
    "encoder": {
        "method": "RATE_FULL_RELATIVE",
        "exposure": 16,
        "tau": 1.0,
        "normalize": True,
    },
}

DEFAULT_TRANSFORMER_HERA_RATE = {
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
        "num_hidden": 2048,
        "num_outputs": 32,
        "probability_in": 0.285,
    },
    "trainer": {
        "epochs": 100,
        "num_nodes": int(os.getenv("NNODES", 1)),
    },
    "encoder": {
        "method": "RATE_FULL",
        "exposure": 4,
        "tau": 1.0,
        "normalize": True,
    },
}

DEFAULT_TRANSFORMER_HERA_LATENCY = {
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
        "num_hidden": 4096,
        "num_outputs": 32,
        "probability_in": 0.382,
    },
    "trainer": {
        "epochs": 100,
        "num_nodes": int(os.getenv("NNODES", 1)),
    },
    "encoder": {
        "method": "LATENCY_FULL",
        "exposure": 2,
        "tau": 1.0,
        "normalize": True,
    },
}

DEFAULT_TRANSFORMER_HERA_DIRECT = {
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
        "num_hidden": 1024,
        "num_outputs": 32,
        "probability_in": 0.230,
    },
    "trainer": {
        "epochs": 100,
        "num_nodes": int(os.getenv("NNODES", 1)),
    },
    "encoder": {
        "method": "DIRECT",
        "exposure": 32,
        "tau": 1.0,
        "normalize": True,
    },
}

DEFAULT_RELU_HERA_RATE = {
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
        "num_hidden": 1024,
        "num_outputs": 32,
        "probability_in": 0.178,
    },
    "trainer": {
        "epochs": 100,
        "num_nodes": int(os.getenv("NNODES", 1)),
    },
    "encoder": {
        "method": "RATE_FULL",
        "exposure": 16,
        "tau": 1.0,
        "normalize": True,
    },
}

DEFAULT_RELU_HERA_DIRECT = {
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
        "num_hidden": 2048,
        "num_outputs": 32,
        "probability_in": 0.149,
    },
    "trainer": {
        "epochs": 100,
        "num_nodes": int(os.getenv("NNODES", 1)),
    },
    "encoder": {
        "method": "DIRECT",
        "exposure": 4,
        "tau": 1.0,
        "normalize": True,
    },
}

DEFAULT_3D_HERA_DIRECT_SINGLE = {
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
        "num_hidden": 16,  # Gets turned into 16 ** 3 = 4096
        "num_outputs": 32,
        "probability_in": 0.25,
    },
    "trainer": {
        "epochs": 100,
        "num_nodes": int(os.getenv("NNODES", 1)),
    },
    "encoder": {
        "method": "DIRECT_SINGLE",
        "exposure": 4,
        "tau": 1.0,
        "normalize": True,
    },
}

DEFAULT_HERA_DIRECT_MEM = {
    "data_source": {
        "data_path": "./data",
        "limit": 1.0,
        "patch_size": 32,
        "stride": 32,
        "dataset": "HERA",
    },
    "dataset": {
        "batch_size": 1,
    },
    "model": {
        "type": "LSM_MEM",
        "num_inputs": 32,
        "num_hidden": 4096,
        "num_outputs": 32,
        "probability_in": 0.25,
    },
    "trainer": {
        "epochs": 100,
        "num_nodes": int(os.getenv("NNODES", 1)),
    },
    "encoder": {
        "method": "DIRECT_MEMBRANE",
        "exposure": 8,
        "tau": 1.0,
        "normalize": True,
    },
}


def get_default_params(
        dataset: str,
        model_type: str,
        delta_normalization: bool = False,
        encoding_method: str = "RATE_FULL",
):
    if dataset == "HERA":
        if model_type == "LSM":
            if encoding_method == "LATENCY_FULL":
                params = copy.deepcopy(DEFAULT_HERA_LATENCY)
            elif encoding_method == "DIRECT":
                params = copy.deepcopy(DEFAULT_HERA_DIRECT)
            else:
                params = copy.deepcopy(DEFAULT_HERA_RATE)
        elif model_type == "LSM_REL":
            params = copy.deepcopy(DEFAULT_HERA_RATE_REL)
        elif model_type == "LSM_TRANSFORMER":
            if encoding_method == "LATENCY_FULL":
                params = copy.deepcopy(DEFAULT_TRANSFORMER_HERA_LATENCY)
            elif encoding_method == "DIRECT":
                params = copy.deepcopy(DEFAULT_TRANSFORMER_HERA_DIRECT)
            else:
                params = copy.deepcopy(DEFAULT_TRANSFORMER_HERA_RATE)
        elif model_type == "LSM_TRANSFORMER_REL":
            params = copy.deepcopy(DEFAULT_TRANSFORMER_HERA_RATE)
        elif model_type == "LSM_RELU":
            if encoding_method == "LATENCY_FULL":
                params = copy.deepcopy(DEFAULT_HERA_LATENCY)
            elif encoding_method == "DIRECT":
                params = copy.deepcopy(DEFAULT_RELU_HERA_DIRECT)
            else:
                params = copy.deepcopy(DEFAULT_HERA_RATE)
        elif model_type == "LSM_RELU_REL":
            params = copy.deepcopy(DEFAULT_HERA_RATE_REL)
        elif model_type == "LSM_3D":
            params = copy.deepcopy(DEFAULT_3D_HERA_DIRECT_SINGLE)
        elif model_type == "LSM_MEM":
            params = copy.deepcopy(DEFAULT_HERA_DIRECT_MEM)
        else:
            raise ValueError(f"Unknown model type {model_type}")
    elif dataset == "LOFAR":
        raise ValueError(f"Unknown model type {model_type}")
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    params["model"]["type"] = model_type
    params["data_source"]["delta_normalization"] = delta_normalization
    return params
