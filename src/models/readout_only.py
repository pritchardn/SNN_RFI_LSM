"""
Readout-only ablation of the LSM models.

The spiking reservoir (and the input projection into it) is removed. The encoded input is fed
directly into the same trainable readout stack that follows the reservoir in `LSM`:
    output_layer (Linear, num_inputs -> num_outputs, no bias)  ->  decoder (linear / relu / transformer)
Encoders, decoding, loss, optimiser and metrics are identical to `LSM`, so any gap between this model
and the matching LSM is attributable to the reservoir.
"""

import lightning.pytorch as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from evaluation import calculate_metrics
from interfaces.data.spiking_data_module import SpikeConverter


class ReadoutOnly(pl.LightningModule):

    def __init__(self, num_inputs: int, num_outputs: int, readout: str = "linear"):
        super().__init__()
        self.converter = None
        self.learning_rate = 1e-4
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.loss = torch.nn.MSELoss()
        self.save_hyperparameters()
        self.output_layer = torch.nn.Linear(num_inputs, num_outputs, bias=False)
        self.readout = readout
        if readout == "linear":
            self.decoder = torch.nn.Linear(num_outputs, num_outputs, bias=False)
        elif readout == "relu":
            self.decoder = torch.nn.ReLU()
        elif readout == "transformer":
            self.decoder = torch.nn.TransformerDecoderLayer(d_model=num_outputs, nhead=4, norm_first=True)
        else:
            raise ValueError("Invalid readout type")

    def forward(self, x):
        x = self.output_layer(x)
        if self.readout == "transformer":
            x = self.decoder(x, x)
        else:
            x = self.decoder(x)
        # Same (output, state, recording) triple as LSM; there is no spiking state here.
        return x, {}, {}

    def training_step(self, batch, batch_idx):
        x, y = batch
        out, _, _ = self(x)
        pred = self.converter.decode_inference_training(out)
        y_true = self.converter.decode_y(y)
        loss = self.loss(pred, y_true)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out, _, _ = self(x)
        pred = self.converter.decode_inference_training(out)
        y_true = self.converter.decode_y(y)
        loss = self.loss(pred, y_true)
        self.log("val_loss", loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        out, _, _ = self(x)
        output_pred = self.converter.decode_inference(out.detach().cpu().numpy())
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
