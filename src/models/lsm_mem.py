import lightning.pytorch as pl
import rockpool.nn.combinators
import torch
from decimal import Decimal, ROUND_HALF_UP

from rockpool.nn.modules import LinearTorch, LIFTorch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from evaluation import calculate_metrics
from interfaces.data.spiking_data_module import SpikeConverter
from models.lsm import generate_sparse_input_weights, initialize_reservoir


class LSMMembrane(pl.LightningModule):

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
        self.decoder = LIFTorch((num_outputs, num_outputs), threshold=1000)  # Only want to integrate, no firing needed
        self.model = rockpool.nn.combinators.Sequential(self.input_layer, self.reservoir, self.output_layer, self.decoder)

    def forward(self, x):
        membranes = []
        self.model.reset_state()
        for i in range(0, x.shape[1], self.converter.exposure):
            segment = x[:, i:i + self.converter.exposure, ...]
            spikes, mem, recording = self.model(segment)
            membranes.append(mem["3_LIFTorch"]["isyn"].clone().requires_grad_(spikes.requires_grad))
        membranes = torch.stack(membranes, dim=1).squeeze().unsqueeze(0)
        return membranes, spikes, recording

    def training_step(self, batch, batch_idx):
        x, y = batch
        membrane, _, _ = self(x)
        pred = self.converter.decode_inference_training(membrane)
        y_true = self.converter.decode_y(y)
        loss = self.loss(pred, y_true)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        membrane, spikes, recording = self(x)
        pred = self.converter.decode_inference_training(membrane)
        y_true = self.converter.decode_y(y)
        loss = self.loss(pred, y_true)
        self.log("val_loss", loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        membrane, _, _ = self(x)
        # Convert output to true output
        output_pred = self.converter.decode_inference(membrane.detach().cpu().numpy())
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