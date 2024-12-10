import abc

import lightning.pytorch as pl
import rockpool.utilities.tree_utils as tu
import torch
from rockpool.nn.combinators import Sequential
from rockpool.nn.modules import LinearTorch, LIFTorch, aLIFTorch
from rockpool.training.torch_loss import make_bounds, bounds_cost
from sklearn.metrics import balanced_accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from evaluation import calculate_metrics
from interfaces.data.spiking_data_module import SpikeConverter

TAU_MEM = 1.0
TAU_SYN = 1.0


def penalized_mse_loss(y_hat, y):
    mse_loss = (y_hat - y) ** 2
    penalty = torch.where((y == 1) * (y_hat < 1.0), 1e5, 1.0)
    penalized_loss = mse_loss * penalty
    return torch.tensor(penalized_loss.sum()).to(y.device)


def focal_loss(y_hat, y):
    gamma = 2
    alpha = 0.25
    loss = -alpha * y * ((1 - y_hat) ** gamma) * torch.log(y_hat) - (1 - alpha) * (1 - y) * (y_hat ** gamma) * torch.log(1 - y_hat)
    return loss.mean().to(y.device)


def dice_loss(y_hat, y):
    smooth = 1e-6
    y_hat = y_hat.view(-1)
    y = y.view(-1)
    intersection = (y * y_hat).sum()
    return 1 - ((2. * intersection + smooth) / (y.sum() + y_hat.sum() + smooth))


def bce_loss(y_hat, y):
    return torch.nn.BCELoss()(y_hat, y)


def output_magnitude_penalty(y_hat, y):
    output_spike_magnitude = torch.log10(torch.abs(y_hat.sum()))
    target_spike_magnitude = torch.log10(y.sum())
    penalty = (target_spike_magnitude - output_spike_magnitude) ** 2
    if torch.isnan(penalty):
        return torch.tensor(10.0).to(y.device)
    return penalty


def defibrillator_loss(y_hat, y):
    # If there are no spikes in y_hat return a massive loss
    if y_hat.sum() == 0:
        return torch.tensor(1e6).to(y.device)
    else:
        return torch.tensor(0.0).to(y.device)


def create_bounds(net):
    lower_bounds, upper_bounds = make_bounds(net.parameters())
    lower_bounds = tu.set_matching(lower_bounds, net.parameters('taus'), 1e-4)
    return lower_bounds, upper_bounds


class BaseModelRockpool(pl.LightningModule):
    def __init__(
            self,
            num_inputs: int,
            num_hidden: int,
            num_outputs: int,
            num_layers: int,
            tau_mem: float,
            tau_syn: float,
            learning_rate: float,
    ):
        super().__init__()
        self.converter = None
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self._init_layers()
        self.loss = dice_loss
        self.learning_rate = learning_rate
        self.lower_bounds, self.upper_bounds = create_bounds(self.model)

    def _init_layers(self):
        layers = []
        if self.num_layers == 1:
            layers.append(
                LinearTorch((self.num_inputs, self.num_hidden), has_bias=False)
            )
            layers.append(
                aLIFTorch(  # TODO: Refinement of parameters
                    self.num_hidden,
                    learning_window=0.2,
                    dt=0.001,
                )
            )
            layers.append(
                LinearTorch((self.num_hidden, self.num_outputs), has_bias=False)
            )
        else:
            for i in range(self.num_layers):
                if i == 0:
                    layers.append(
                        aLIFTorch(  # TODO: Refinement of parameters
                            self.num_inputs,
                            learning_window=0.2,
                            dt=0.001,
                        )
                    )
                    layers.append(
                        LinearTorch((self.num_inputs, self.num_hidden), has_bias=False)
                    )
                elif i == self.num_layers - 1:
                    layers.append(
                        LinearTorch((self.num_hidden, self.num_outputs), has_bias=False)
                    )

                else:
                    layers.append(
                        aLIFTorch(  # TODO: Refinement of parameters
                            self.num_hidden,
                            learning_window=0.2,
                            dt=0.001,
                        )
                    )
                    if not i == self.num_layers - 2:
                        layers.append(
                            LinearTorch((self.num_hidden, self.num_hidden), has_bias=False)
                        )
        print(layers)
        self.model = Sequential(*layers)
        self.model = self.model.to(self.device)
        print(self.model)

    def set_converter(self, converter: SpikeConverter):
        self.converter = converter

    def calc_accuracy(self, y_hat, y):
        score = balanced_accuracy_score(y_hat.flatten(), y.flatten())
        self.log("accuracy", score)

    @abc.abstractmethod
    def forward(self, x):
        pass

    def calc_loss(self, y_hat, y):
        constraint_cost = bounds_cost(self.model.parameters(), self.lower_bounds, self.upper_bounds)
        metric_loss = self.loss(y_hat, y)
        penalty_loss = output_magnitude_penalty(y_hat, y)
        defib_loss = defibrillator_loss(y_hat, y)
        if torch.isnan(metric_loss):
            metric_loss = torch.tensor(y.nelement()).to(self.device)
        loss = metric_loss + y.nelement() * constraint_cost + penalty_loss + defib_loss
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        spike_hat, mem_hat = self(x)
        loss = self.calc_loss(spike_hat, y)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        spike_hat, mem_hat = self(x)
        loss = self.calc_loss(spike_hat, y)
        self.log("val_loss", loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        spike_hat, mem_hat = self(x)
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


class LitModelRockpool(BaseModelRockpool):
    def __init__(
            self,
            num_inputs: int,
            num_hidden: int,
            num_outputs: int,
            num_layers: int,
            tau_mem: float,
            tau_syn: float,
            learning_rate: float,
    ):
        super().__init__(
            num_inputs,
            num_hidden,
            num_outputs,
            num_layers,
            tau_mem,
            tau_syn,
            learning_rate,
        )

    def forward(self, x):
        self.model.reset_state()
        spike, mem, recording = self.model(x)
        return spike, recording


class LitModelPatchedRockpool(BaseModelRockpool):
    def __init__(
            self,
            num_inputs: int,
            num_hidden: int,
            num_outputs: int,
            num_layers: int,
            tau_mem: float,
            tau_syn: float,
            learning_rate: float,
    ):
        super().__init__(
            num_inputs,
            num_hidden,
            num_outputs,
            num_layers,
            tau_mem,
            tau_syn,
            learning_rate,
        )

    def forward(self, x):
        self.model.reset_state()
        data = x.view(*(x.shape[:-2]), -1).squeeze(2)
        spike, _, recording = self.model(data)
        spike = spike.view(*(spike.shape[:-1]), -1, x.shape[-1])
        full_spike = spike.unsqueeze(2)
        return full_spike, recording
