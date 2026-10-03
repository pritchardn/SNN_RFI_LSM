"""
Main script to run the experiment
"""

import os

from config import get_default_params
from experiment import Experiment


def main():
    model_type = os.getenv("MODEL_TYPE", "LSM")
    dataset = os.getenv("DATASET", "HERA")
    num_layers = int(os.getenv("NUM_LAYERS", 2))
    encoder = os.getenv("ENCODER_METHOD", "RATE_FULL")
    tau_mem = os.getenv("TAU_MEM", None)
    tau_mem = float(tau_mem) if tau_mem else None
    tau_syn = os.getenv("TAU_SYN", None)
    tau_syn = float(tau_syn) if tau_syn else None
    plot = os.getenv("PLOT", False) == "True"
    delta_normalization = os.getenv("DELTA_NORMALIZATION", False) == "True"
    config = get_default_params(
        dataset, model_type, delta_normalization, encoding_method=encoder
    )
    config["encoder"]["method"] = encoder
    config["data_source"]["data_path"] = os.getenv(
        "DATA_PATH", config["data_source"]["data_path"]
    )
    config["model"]["num_hidden"] = int(os.getenv("NUM_HIDDEN", config["model"]["num_hidden"]))
    config["model"]["num_layers"] = num_layers
    if tau_mem:
        config["model"]["tau_mem"] = tau_mem
    if tau_syn:
        config["model"]["tau_syn"] = tau_syn

    config["data_source"]["limit"] = float(
        os.getenv("LIMIT", config["data_source"]["limit"])
    )
    config["encoder"]["exposure"] = int(
        os.getenv("EXPOSURE", config["encoder"].get("exposure", 1))
    )
    config["dataset"]["batch_size"] = int(
        os.getenv("BATCH_SIZE", config["dataset"]["batch_size"])
    )
    config["trainer"]["epochs"] = int(os.getenv("EPOCHS", config["trainer"]["epochs"]))
    root_dir = os.getenv("OUTPUT_DIR", "./")
    print(config)
    experiment = Experiment(root_dir=root_dir)
    experiment.from_config(config)
    experiment.prepare()
    print("Preparation complete")
    experiment.train()
    print("Training complete")
    experiment.evaluate(plot)
    print("Evaluation complete")
    experiment.save_model()


if __name__ == "__main__":
    main()
