import lightning.pytorch as pl
import rockpool.nn.combinators
import torch
from decimal import Decimal, ROUND_HALF_UP
from rockpool.nn.modules import LinearTorch, aLIFTorch
from rockpool.weights.reservoirweights import rndm_ei_net
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

from evaluation import calculate_metrics
from interfaces.data.spiking_data_module import SpikeConverter
from plotting import plot_input_raster, plot_example_mask, plot_target_raster, \
    plot_network_internals, plot_example_inference


def initialize_taus(num_exc, num_inh, exposure):
    taus = torch.zeros(num_exc + num_inh)
    taus[:num_exc] = exposure / 3
    taus[num_exc:] = exposure
    return taus


def initialize_reservoir(num_exc, num_inh, exposure):
    reservoir_weights = rndm_ei_net(num_exc, num_inh)
    taus = initialize_taus(num_exc, num_inh, exposure)
    reservoir = aLIFTorch(num_exc + num_inh, learning_window=0.2, dt=0.001, tau_mem=taus)
    reservoir.w_rec = torch.tensor(reservoir_weights).float()
    reservoir.w_ahp.requires_grad = False
    return reservoir


class LSM(pl.LightningModule):

    def __init__(self, num_inputs: int, num_hidden: int, num_outputs: int, exposure: int):
        super().__init__()
        self.converter = None
        self.learning_rate = 1e-4
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.loss = torch.nn.MSELoss()
        self.save_hyperparameters()
        self.input_layer = LinearTorch((num_inputs, num_hidden))
        self.output_layer = LinearTorch((num_hidden, num_outputs))
        self.reservoir = initialize_reservoir(int(Decimal(num_hidden * 0.8).to_integral(rounding=ROUND_HALF_UP)),
                                              int(Decimal(num_hidden * 0.2).to_integral(rounding=ROUND_HALF_UP)),
                                              exposure)
        self.reservoir.w_ahp.requires_grad = False
        self.model = rockpool.nn.combinators.Sequential(self.input_layer, self.reservoir, self.output_layer)

    def forward(self, x):
        x, mem, recording = self.model(x)
        return x, mem, recording

    def training_step(self, batch, batch_idx):
        x, y = batch
        spike_hat, _, _ = self(x)
        loss = self.loss(spike_hat, y)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        spike_hat, mem, recording = self(x)
        loss = self.loss(spike_hat, y)
        self.log("val_loss", loss, sync_dist=True)
        if batch_idx == 0 and self.trainer.local_rank == 0:
            # Reshape spike_hat to [N, T, C, F T]
            spike_hat_plot = torch.reshape(
                spike_hat.detach().cpu(),
                (spike_hat.shape[0], -1, spike_hat.shape[-1], spike_hat.shape[-1]),
            )
            spike_hat_plot = torch.moveaxis(spike_hat_plot, -2, -1)
            plot_example_inference(
                spike_hat_plot[0, :, ::],
                str(self.current_epoch),
                self.trainer.log_dir,
            )
            plot_input_raster(
                x[0, ::].detach().cpu().numpy(),
                self.trainer.current_epoch,
                self.trainer.log_dir,
            )
            target = y[0, ::].detach().cpu().numpy()
            plot_target_raster(target, self.trainer.current_epoch, self.trainer.log_dir)
            target = y[0, :: y[0].shape[0] // y[0].shape[-1], :].detach().cpu().numpy()
            target = np.moveaxis(target, -2, -1)
            plot_example_mask(
                np.expand_dims(target, axis=-1),
                str(self.current_epoch),
                self.trainer.log_dir,
            )
            try:
                plot_network_internals(
                    self.model, recording, self.trainer.current_epoch, self.trainer.log_dir
                )
            except ValueError as e:
                print(e)

    def test_step(self, batch, batch_idx):
        x, y = batch
        spike_hat, _, _ = self(x)
        # Convert output to true output
        output_pred = self.converter.decode_inference(spike_hat.detach().cpu().numpy())
        y_true = self.converter.decode_y(y.detach().cpu().numpy())
        accuracy, mse, auroc, auprc, f1 = calculate_metrics(y_true, output_pred)
        self.log("test_accuracy", accuracy, sync_dist=True)
        self.log("test_mse", mse, sync_dist=True)
        self.log("test_auroc", auroc, sync_dist=True)
        self.log("test_auprc", auprc, sync_dist=True)
        self.log("test_f1", f1, sync_dist=True)
        return accuracy, mse, auroc, auprc, f1

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def set_converter(self, converter: SpikeConverter):
        self.converter = converter
