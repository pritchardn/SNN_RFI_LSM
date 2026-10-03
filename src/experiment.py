"""
This module provides a class to manage experiments with the PyTorch Lightning framework.
It is in charge of loading the configuration, setting up the data, model, and trainer,
and fitting the model.
"""

import glob
import json
import os

import lightning.pytorch as pl
import torch

from data.data_loaders import (
    HeraDataLoader,
    LofarDataLoader,
    HeraDeltaNormLoader,
    LofarDeltaNormLoader,
)
from data.data_module import ConfiguredDataModule
from data.data_module_builder import DataModuleBuilder
from data.spike_converters import (
    LatencySpikeConverter,
    LatencyFullSpikeConverter,
    RateSpikeConverter,
    RateFullBalancedSpikeConverter,
    RateFullSpikeConverter,
    DeltaSpikeConverter,
    DeltaExposureSpikeConverter,
    ForwardStepConverter,
    NonConverter,
    RateFullRelativeSpikeConverter,
    DirectSpikeConverter,
    DirectSingleStepConverter,
    LatencyFullRelativeSpikeConverter,
    DirectMembraneConverter,
)
from data.utils import reconstruct_patches
from models.lsm import LSM, LSM3D
from models.lsm_mem import LSMMembrane
from models.readout_only import ReadoutOnly
from evaluation import final_evaluation
from interfaces.data.raw_data_loader import RawDataLoader
from interfaces.data.spiking_data_module import SpikeConverter

READOUT_ONLY_MODELS = {
    "READOUT_LINEAR": "linear",
    "READOUT_LINEAR_REL": "linear",
    "READOUT_RELU": "relu",
    "READOUT_RELU_REL": "relu",
    "READOUT_TRANSFORMER": "transformer",
    "READOUT_TRANSFORMER_REL": "transformer",
}


def data_source_from_config(config: dict) -> RawDataLoader:
    data_path = config.get("data_path")
    patch_size = config.get("patch_size")
    stride = config.get("stride")
    limit = config.get("limit")
    dataset = config.get("dataset")
    delta_normalization = config.get("delta_normalization")
    if dataset == "HERA":
        if delta_normalization:
            data_source = HeraDeltaNormLoader(
                data_path, patch_size=patch_size, stride=stride, limit=limit
            )
        else:
            data_source = HeraDataLoader(
                data_path, patch_size=patch_size, stride=stride, limit=limit
            )
    elif dataset == "LOFAR":
        if delta_normalization:
            data_source = LofarDeltaNormLoader(
                data_path, patch_size=patch_size, stride=stride, limit=limit
            )
        else:
            data_source = LofarDataLoader(
                data_path, patch_size=patch_size, stride=stride, limit=limit
            )
    else:
        raise NotImplementedError(f"Dataset {dataset} is not supported.")
    return data_source


def dataset_from_config(
    config: dict, data_source: RawDataLoader, encoder: SpikeConverter
) -> ConfiguredDataModule:
    batch_size = config.get("batch_size")
    data_builder = DataModuleBuilder()
    data_builder.set_dataset(data_source)
    data_builder.set_encoding(encoder)
    dataset = data_builder.build(batch_size)
    return dataset


def model_from_config(config: dict, exposure: int) -> pl.LightningModule:
    model_type = config.get("type")
    num_inputs = config.get("num_inputs")
    num_hidden = config.get("num_hidden")
    num_outputs = config.get("num_outputs")
    p_in = config.get("probability_in")
    if model_type == "LSM" or model_type == "LSM_REL":
        model = LSM(
            num_inputs=num_inputs,
            num_hidden=num_hidden,
            num_outputs=num_outputs,
            exposure=exposure,
            p_in=p_in
        )
        return model
    elif model_type == "LSM_RELU" or model_type == "LSM_RELU_REL":
        model = LSM(
            num_inputs=num_inputs,
            num_hidden=num_hidden,
            num_outputs=num_outputs,
            exposure=exposure,
            p_in=p_in,
            readout="relu",
        )
        return model
    elif model_type == "LSM_TRANSFORMER" or model_type == "LSM_TRANSFORMER_REL":
        model = LSM(
            num_inputs=num_inputs,
            num_hidden=num_hidden,
            num_outputs=num_outputs,
            exposure=exposure,
            p_in=p_in,
            readout="transformer",
        )
        return model
    if model_type in READOUT_ONLY_MODELS:
        return ReadoutOnly(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            readout=READOUT_ONLY_MODELS[model_type],
        )
    if model_type == "LSM_MEM":
        model = LSMMembrane(
            num_inputs=num_inputs,
            num_hidden=num_hidden,
            num_outputs=num_outputs,
            exposure=exposure,
            p_in=p_in
        )
        return model
    elif model_type == "LSM_3D":
        model = LSM3D(
            num_inputs=num_inputs,
            num_hidden=num_hidden,
            num_outputs=num_outputs,
            exposure=exposure,
            p_in=p_in
        )
        return model
    else:
        raise NotImplementedError(f"Model type {model_type} is not supported.")


def trainer_from_config(config: dict, root_dir: str, callbacks=None) -> pl.Trainer:
    if callbacks is None:
        callbacks = []
    num_gpus = torch.cuda.device_count()
    epochs = config.get("epochs")
    # patience = config.get("patience", 10)
    # early_stopping_callback = pl.callbacks.EarlyStopping(
    #     monitor="val_loss", mode="min", patience=patience, min_delta=1e-4
    # )
    if num_gpus > 0:
        trainer = pl.trainer.Trainer(
            max_epochs=epochs,
            benchmark=True,
            default_root_dir=root_dir,
            devices=num_gpus,
            num_nodes=config.get("num_nodes", 1),
            callbacks=callbacks,
            log_every_n_steps=25,
        )
    else:
        if torch.mps.is_available():
            accelerator = "mps"
        else:
            accelerator = "cpu"
        trainer = pl.trainer.Trainer(
            max_epochs=epochs,
            benchmark=True,
            default_root_dir=root_dir,
            num_nodes=config.get("num_nodes", 1),
            accelerator=accelerator,
            callbacks=callbacks,
            log_every_n_steps=8,
        )
    return trainer


def encoder_from_config(config: dict) -> SpikeConverter:
    encoder = None
    if config.get("method") == "LATENCY":
        exposure = config.get("exposure")
        tau = config.get("tau")
        normalize = config.get("normalize")
        encoder = LatencySpikeConverter(exposure=exposure, tau=tau, normalize=normalize)
    elif config.get("method") == "LATENCY_FULL":
        exposure = config.get("exposure")
        tau = config.get("tau")
        normalize = config.get("normalize")
        encoder = LatencyFullSpikeConverter(
            exposure=exposure, tau=tau, normalize=normalize
        )
    elif config.get("method") == "LATENCY_FULL_RELATIVE":
        exposure = config.get("exposure")
        tau = config.get("tau")
        normalize = config.get("normalize")
        encoder = LatencyFullRelativeSpikeConverter(
            exposure=exposure, tau=tau, normalize=normalize
        )
    elif config.get("method") == "RATE":
        exposure = config.get("exposure")
        encoder = RateSpikeConverter(exposure=exposure)
    elif config.get("method") == "RATE_FULL":
        exposure = config.get("exposure")
        encoder = RateFullSpikeConverter(exposure=exposure)
    elif config.get("method") == "RATE_FULL_BALANCED":
        exposure = config.get("exposure")
        encoder = RateFullBalancedSpikeConverter(exposure=exposure)
    elif config.get("method") == "RATE_FULL_RELATIVE":
        exposure = config.get("exposure")
        encoder = RateFullRelativeSpikeConverter(exposure=exposure)
    elif config.get("method") == "DELTA":
        threshold = config.get("threshold")
        off_spikes = config.get("off_spikes")
        encoder = DeltaSpikeConverter(threshold=threshold, off_spikes=off_spikes)
    elif config.get("method") == "DELTA_EXPOSURE":
        threshold = config.get("threshold")
        exposure = config.get("exposure")
        encoder = DeltaExposureSpikeConverter(threshold=threshold, exposure=exposure)
    elif config.get("method") == "FORWARDSTEP":
        threshold = config.get("threshold")
        exposure = config.get("exposure")
        tau = config.get("tau")
        normalize = config.get("normalize")
        exposure_mode = config.get("exposure_mode")
        encoder = ForwardStepConverter(
            threshold=threshold,
            exposure=exposure,
            tau=tau,
            normalize=normalize,
            exposure_mode=exposure_mode,
        )
    elif config.get("method") == "DIRECT":
        encoder = DirectSpikeConverter(
            exposure=config.get("exposure"),
        )
    elif config.get("method") == "DIRECT_SINGLE":
        encoder = DirectSingleStepConverter(
            exposure=config.get("exposure"),
        )
    elif config.get("method") == "DIRECT_MEMBRANE":
        encoder = DirectMembraneConverter(
            exposure=config.get("exposure"),
        )
    elif config.get("method") == "ANN":
        encoder = NonConverter()
    elif config.get("method") == "ANN_PATCHED":
        encoder = NonConverter(patched=True)
    return encoder


class Experiment:
    def __init__(self, root_dir="./", callbacks=None):
        self.data_source = None
        self.dataset = None
        self.model = None
        self.trainer = None
        self.encoder = None
        self.configuration = None
        self.ready = False
        self.checkpoint_path = None
        self.root_dir = root_dir
        self.callbacks = callbacks

    def from_config(self, config: dict):
        self.configuration = config
        if self.configuration.get("encoder"):
            self.encoder = encoder_from_config(config.get("encoder"))
        if self.configuration.get("data_source"):
            self.data_source = data_source_from_config(config.get("data_source"))
            if self.configuration.get("dataset") and self.encoder:
                self.dataset = dataset_from_config(
                    config.get("dataset"), self.data_source, self.encoder
                )
        if self.configuration.get("model"):
            self.model = model_from_config(config.get("model"), config.get("encoder").get("exposure"))
        if self.configuration.get("trainer"):
            self.trainer = trainer_from_config(
                config.get("trainer"), self.root_dir, callbacks=self.callbacks
            )

    def from_checkpoint(self, working_dir: str):
        self.load_config(os.path.join(working_dir, "config.json"))
        checkpoint_dir = os.path.join(working_dir, "checkpoints", "*.ckpt")
        checkpoint_file = glob.glob(checkpoint_dir)[0]
        self.checkpoint_path = checkpoint_file
        self.from_config(self.configuration)

    def add_dataset(self, data_source: RawDataLoader):
        self.data_source = data_source
        self.dataset = None
        self.ready = False

    def save_config(self):
        out_dir = self.trainer.log_dir
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.json"), "w") as ofile:
            json.dump(self.configuration, ofile, indent=4)

    def load_config(self, config_path: str):
        with open(config_path, "r") as ifile:
            self.configuration = json.load(ifile)

    def save_model(self):
        try:
            out_dir = self.trainer.log_dir
            os.makedirs(out_dir, exist_ok=True)
            input_sample, _ = next(iter(self.dataset.test_dataloader()))
            self.model.to_onnx(
                os.path.join(out_dir, "model.onnx"), input_sample, export_params=True
            )
        except Exception as e:
            print(f"Error during model saving: {e}")

    def prepare(self):
        err_msg = ""
        if not self.ready:
            if self.data_source and self.encoder:
                if not self.dataset:
                    self.dataset = dataset_from_config(
                        self.configuration.get("dataset"),
                        self.data_source,
                        self.encoder,
                    )
            else:
                err_msg += "Data source not set.\n"
            if not self.model:
                err_msg += "Model not set.\n"
            if not self.trainer:
                err_msg += "Trainer not set.\n"
            else:
                if self.trainer.global_rank == 0:
                    self.save_config()
            if not self.encoder:
                err_msg += "Encoder not set.\n"
            else:
                self.model.set_converter(self.encoder)
            if err_msg != "":
                raise ValueError(err_msg)
            self.ready = True

    def train(self):
        if not self.ready:
            raise RuntimeError("Experiment not ready.")
        self.model.train()
        if self.checkpoint_path:
            self.trainer.fit(self.model, self.dataset, ckpt_path=self.checkpoint_path)
        else:
            self.trainer.fit(self.model, self.dataset)

    def evaluate(self, plot=False):
        self.model.eval()
        metrics = self.trainer.test(self.model, self.dataset.test_dataloader())
        accuracy = metrics[0]["test_accuracy"]
        mse = metrics[0]["test_mse"]
        auroc = metrics[0]["test_auroc"]
        auprc = metrics[0]["test_auprc"]
        f1 = metrics[0]["test_f1"]
        output = json.dumps(
            {
                "accuracy": accuracy,
                "mse": mse,
                "auroc": auroc,
                "auprc": auprc,
                "f1": f1,
            }
        )
        # Write output
        with open(os.path.join(self.trainer.log_dir, "metrics.json"), "w") as ofile:
            json.dump(output, ofile, indent=4)
        if plot and self.trainer.local_rank == 0:
            try:
                mask_orig = reconstruct_patches(
                    self.data_source.fetch_test_y(),
                    self.data_source.original_size,
                    self.data_source.stride,
                )
                original_data = reconstruct_patches(
                    self.data_source.fetch_test_x(),
                    self.data_source.original_size,
                    self.data_source.stride,
                )
                final_evaluation(
                    self.model,
                    self.dataset,
                    self.encoder,
                    original_data,
                    mask_orig,
                    self.data_source.fetch_test_x(),
                    self.data_source.fetch_test_y(),
                    self.encoder.exposure,
                    self.trainer.log_dir,
                    self.configuration["model"]["type"],
                )
            except RuntimeError as e:
                print(f"Error during evaluation: {e}")
        return accuracy, mse, auroc, auprc, f1
