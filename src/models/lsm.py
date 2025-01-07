import math

import lightning.pytorch as pl
import rockpool.nn.combinators
import torch
from decimal import Decimal, ROUND_HALF_UP
from rockpool.nn.modules import LinearTorch, LIFTorch
from rockpool.weights.reservoirweights import rndm_ei_net
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from evaluation import calculate_metrics
from interfaces.data.spiking_data_module import SpikeConverter
from plotting import plot_input_raster, plot_example_mask, plot_target_raster, \
    plot_network_internals, plot_example_inference

BASE_TAU = 0.002

def initialize_taus(num_exc, num_inh, exposure):
    taus_exc = 0.001 + torch.rand(num_exc) * (0.01 - 0.001)
    taus_inh = 0.001 + torch.rand(num_inh) * (0.01 - 0.001)
    taus = torch.cat((taus_exc, taus_inh))
    return taus


def initialize_reservoir(num_exc, num_inh, exposure):
    reservoir_weights = rndm_ei_net(num_exc, num_inh)
    taus = initialize_taus(num_exc, num_inh, exposure)
    taus_syns = initialize_taus(num_exc, num_inh, exposure)
    reservoir = LIFTorch(num_exc + num_inh, learning_window=0.2, dt=0.001, tau_mem=taus, tau_syn=taus_syns)
    reservoir.w_rec = torch.tensor(reservoir_weights).float()
    reservoir.bias.requires_grad = False
    reservoir.tau_mem.requires_grad = False
    reservoir.tau_syn.requires_grad = False
    reservoir.threshold.requires_grad = False
    return reservoir


def euclidean_distance(p1, p2):
    x_i, x_j, x_k = p1
    y_i, y_j, y_k = p2
    distance = math.sqrt((y_i - x_i) ** 2 + (y_j - x_j) ** 2 + (y_k - x_k) ** 2)
    return distance


def connection_probability(p1, p2, conn_strength, sparsity=2.0):
    return conn_strength * math.e ** (-(euclidean_distance(p1, p2)/sparsity) ** 2)


def ind_to_point(dim, ind) -> (int, int, int):
    yz = ind % (dim ** 2)
    x = ind // (dim ** 2)
    y = yz // dim
    z = yz % dim
    return x, y, z


def select_conn_size(n1_type, n2_type):
    if n1_type and n2_type:  # EE
        return 0.3
    if n1_type and not n2_type:  # EI
        return 0.2
    if not n1_type and n2_type:  # IE
        return 0.4
    if not n1_type and not n2_type:  # II
        return 0.1


def generate_conn_weight(n1_type, n2_type):
    rand = torch.rand(())
    if n1_type and n2_type:  # EE
        rand *= 0.6
    if n1_type and not n2_type:  # EI
        rand *= 0.4
    if not n1_type and n2_type:  # IE
        rand *= -1.2
    if not n1_type and not n2_type:  # II
        rand *= -0.4
    return rand


def generate_reservoir_3d_weights(dim, exc_inh_ratio):
    num_neurons = dim ** 3
    reservoir_weights = torch.zeros(num_neurons, num_neurons)
    neuron_type_exc = torch.where(torch.rand(dim, dim, dim) >= exc_inh_ratio, True, False)
    # 1.0 for excitory, 0 for inhibitory
    for i in tqdm(range(num_neurons)):
        for j in range(num_neurons):
            p1 = ind_to_point(dim, i)
            p2 = ind_to_point(dim, j)
            conn_size = select_conn_size(neuron_type_exc[p1], neuron_type_exc[p2])
            if connection_probability(p1, p2, conn_size) < torch.rand(()):
                reservoir_weights[i] = generate_conn_weight(neuron_type_exc[p1], neuron_type_exc[p2])
    return reservoir_weights


def initialize_reservoir_3d(dim, exc_inh_ratio=0.8):
    num_neurons = dim ** 3
    weights = generate_reservoir_3d_weights(dim, exc_inh_ratio)
    reservoir = LIFTorch(num_neurons, learning_window=0.2, dt=0.001)
    reservoir.w_rec = torch.tensor(weights).float()
    reservoir.bias.requires_grad = False
    reservoir.tau_mem.requires_grad = False
    reservoir.tau_syn.requires_grad = False
    reservoir.threshold.requires_grad = False
    return reservoir


class LSM(pl.LightningModule):

    def __init__(self, num_inputs: int, num_hidden: int, num_outputs: int, exposure: int, p_in: float, plot=False):
        super().__init__()
        self.converter = None
        self.plot = plot
        self.learning_rate = 1e-4
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.p_in = p_in
        self.loss = torch.nn.MSELoss()
        self.save_hyperparameters()
        self.input_layer = LinearTorch((num_inputs, num_hidden))
        self.input_layer.weight = torch.nn.Parameter(generate_sparse_input_weights(self.num_inputs, self.num_hidden, p_in))
        self.output_layer = LinearTorch((num_hidden, num_outputs))
        self.reservoir = initialize_reservoir(int(Decimal(num_hidden * 0.8).to_integral(rounding=ROUND_HALF_UP)),
                                              int(Decimal(num_hidden * 0.2).to_integral(rounding=ROUND_HALF_UP)),
                                              exposure)
        self.decoder = torch.nn.TransformerDecoderLayer(d_model=num_outputs, nhead=4)
        self.model = rockpool.nn.combinators.Sequential(self.input_layer, self.reservoir, self.output_layer)

    def forward(self, x):
        x, mem, recording = self.model(x)
        x = self.decoder(x, x)
        return x, mem, recording

    def training_step(self, batch, batch_idx):
        x, y = batch
        spike_hat, _, _ = self(x)
        pred = self.converter.decode_inference_training(spike_hat)
        y_true = self.converter.decode_y(y)
        loss = self.loss(pred, y_true)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        spike_hat, mem, recording = self(x)
        pred = self.converter.decode_inference_training(spike_hat)
        y_true = self.converter.decode_y(y)
        loss = self.loss(pred, y_true)
        self.log("val_loss", loss, sync_dist=True)
        if batch_idx == 0 and self.trainer.local_rank == 0 and self.plot:
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
        y_true = y.detach().cpu().numpy()
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


def generate_sparse_input_weights(num_inputs, num_hidden, p_in: float = 0.1):
    weights = torch.zeros(num_inputs, num_hidden)
    mask = torch.rand(num_inputs, num_hidden) < p_in
    weights[mask] = (torch.rand(mask.sum()) * 0.2) + 0.2  # [0.2, 0.4]
    return weights


class LSM3D(pl.LightningModule):
    def __init__(self, num_inputs: int, num_hidden: int, num_outputs: int, exposure: int, p_in: float, plot=False):
        super().__init__()
        self.converter = None
        self.plot = plot
        self.learning_rate = 1e-4
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden ** 3
        self.num_outputs = num_outputs
        self.exposure = exposure
        self.p_in = p_in
        self.loss = torch.nn.MSELoss()
        self.save_hyperparameters()
        self.input_layer = LinearTorch((self.num_inputs, self.num_hidden))
        self.input_layer.weight = torch.nn.Parameter(generate_sparse_input_weights(self.num_inputs, self.num_hidden, p_in))
        self.input_layer.requires_grad = False
        self.output_layer = LinearTorch((self.num_hidden, num_outputs))
        self.reservoir = initialize_reservoir_3d(num_hidden)
        self.model = rockpool.nn.combinators.Sequential(self.input_layer, self.reservoir, self.output_layer)

    def forward(self, x):
        x, mem, recording = self.model(x)
        return x, mem, recording

    def training_step(self, batch, batch_idx):
        x, y = batch
        spike_hat, _, _ = self(x)
        pred = self.converter.decode_inference_training(spike_hat)
        y_true = self.converter.decode_y(y)
        loss = self.loss(pred, y_true)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        spike_hat, mem, recording = self(x)
        pred = self.converter.decode_inference_training(spike_hat)
        y_true = self.converter.decode_y(y)
        loss = self.loss(pred, y_true)
        self.log("val_loss", loss, sync_dist=True)
        if batch_idx == 0 and self.trainer.local_rank == 0 and self.plot:
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

    def test_step(self, batch, batch_idx):
        x, y = batch
        spike_hat, _, _ = self(x)
        # Convert output to true output
        output_pred = self.converter.decode_inference(spike_hat.detach().cpu().numpy())
        y_true = y.detach().cpu().numpy()
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