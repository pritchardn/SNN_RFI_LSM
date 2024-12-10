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
        "type": "FC_RATE",
        "num_inputs": 32,
        "num_hidden": 128,
        "num_outputs": 32,
        "num_layers": 2,
        "beta": 0.599215428763344,
        "tau_mem": 1.0,
        "tau_syn": 1.0,
        "learning_rate": 1e-3,
    },
    "trainer": {
        "epochs": 50,
        "num_nodes": int(os.getenv("NNODES", 1)),
    },
    "encoder": {
        "method": "RATE",
        "exposure": 16,
        "tau": 1.0,
        "normalize": True,
    },
}


DEFAULT_HERA_ANN = {
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
        "type": "FC_ANN",
        "num_inputs": 32,
        "num_hidden": 128,
        "num_outputs": 32,
        "num_layers": 2,
        "learning_rate": 1e-3,
    },
    "trainer": {
        "epochs": 100,
        "num_nodes": int(os.getenv("NNODES", 1)),
    },
    "encoder": {
        "method": "ANN",
    },
}

DEFAULT_HERA_ANN_DIVNORM = copy.deepcopy(DEFAULT_HERA_ANN)
DEFAULT_HERA_ANN_DIVNORM["model"]["num_hidden"] = 512
DEFAULT_HERA_ANN_DIVNORM["model"]["num_layers"] = 3
DEFAULT_HERA_ANN_DIVNORM["data_source"]["delta_normalization"] = True

DEFAULT_LOFAR_ANN = {
    "data_source": {
        "data_path": "./data",
        "limit": 1.0,
        "patch_size": 32,
        "stride": 32,
        "dataset": "LOFAR",
    },
    "dataset": {
        "batch_size": 36,
    },
    "model": {
        "type": "FC_ANN",
        "num_inputs": 32,
        "num_hidden": 512,
        "num_outputs": 32,
        "num_layers": 6,
        "learning_rate": 1e-3,
    },
    "trainer": {
        "epochs": 100,
        "num_nodes": int(os.getenv("NNODES", 1)),
    },
    "encoder": {
        "method": "ANN",
    },
}

DEFAULT_LOFAR_ANN_DIVNORM = copy.deepcopy(DEFAULT_LOFAR_ANN)
DEFAULT_LOFAR_ANN_DIVNORM["model"]["num_hidden"] = 512
DEFAULT_LOFAR_ANN_DIVNORM["model"]["num_layers"] = 6
DEFAULT_LOFAR_ANN_DIVNORM["data_source"]["delta_normalization"] = True


def get_default_params(
    dataset: str,
    model_type: str,
    model_size: int = 128,
    exposure_mode: str = None,
    delta_normalization: bool = False,
):
    if dataset == "HERA":
        if model_type == "FC_ANN":
            if delta_normalization:
                params = DEFAULT_HERA_ANN_DIVNORM
            else:
                params = DEFAULT_HERA_ANN
        elif model_type == "FC_LATENCY_ROCKPOOL":
            params = copy.deepcopy(DEFAULT_HERA_LATENCY)
            params["model"]["type"] = "FC_LATENCY_ROCKPOOL"
            params["encoder"]["method"] = "LATENCY_FULL"
            params["data_source"]["stride"] = 16
            params["data_source"]["patch_size"] = 16
            params["model"]["num_inputs"] = 16
            params["model"]["num_outputs"] = 16
            params["model"]["num_hidden"] = 256
        elif model_type == "FC_RATE_ROCKPOOL":
            params = copy.deepcopy(DEFAULT_HERA_RATE)
            params["encoder"]["method"] = "RATE_FULL"
            params["data_source"]["stride"] = 16
            params["data_source"]["patch_size"] = 16
            params["model"]["num_inputs"] = 16
            params["model"]["num_outputs"] = 16
            params["model"]["num_hidden"] = 256
        elif model_type == "FCP_LATENCY_ROCKPOOL":
            params = copy.deepcopy(DEFAULT_HERA_LATENCY)
            params["model"]["type"] = "FCP_LATENCY_ROCKPOOL"
            params["encoder"]["method"] = "LATENCY_FULL"
            stride = params["data_source"]["stride"]
            params["model"]["num_inputs"] = stride * stride
            params["model"]["num_outputs"] = stride * stride
            params["model"]["num_hidden"] = max(model_size, stride * stride)
        else:
            raise ValueError(f"Unknown model type {model_type}")
    elif dataset == "LOFAR":
        if model_type == "FC_ANN":
            if delta_normalization:
                params = DEFAULT_LOFAR_ANN_DIVNORM
            else:
                params = DEFAULT_LOFAR_ANN
        elif model_type == "FC_RATE_ROCKPOOL":
            params = copy.deepcopy(DEFAULT_HERA_RATE)
            params["data_source"]["dataset"] = "LOFAR"
            params["encoder"]["method"] = "RATE_FULL"
            params["data_source"]["stride"] = 16
            params["data_source"]["patch_size"] = 16
            params["model"]["num_inputs"] = 16
            params["model"]["num_outputs"] = 16
        else:
            raise ValueError(f"Unknown model type {model_type}")
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    params["model"]["type"] = model_type
    params["data_source"]["delta_normalization"] = delta_normalization
    return params
