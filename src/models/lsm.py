import rockpool.nn.losses
import torch
import lightning.pytorch as pl
from rockpool.nn.modules import LinearTorch, aLIFTorch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from evaluation import calculate_metrics
from interfaces.data.spiking_data_module import SpikeConverter


class LSM(pl.LightningModule):
    def __init__(self, num_inputs: int, num_hidden: int, num_outputs: int):
        super().__init__()
        self.converter = None
        self.learning_rate = 1e-4
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.loss = torch.nn.MSELoss()
        self.save_hyperparameters()
        self.input_layer = LinearTorch((num_inputs, num_hidden))
        self.input_layer.weight.requires_grad = False
        self.output_layer = LinearTorch((num_hidden, num_outputs))
        self.reservoir = aLIFTorch(num_hidden, learning_window=0.2, dt=0.001)
        self.reservoir.w_ahp.requires_grad = False

    def forward(self, x):
        x = self.input_layer(x)
        x = self.reservoir(x)
        x = self.output_layer(x)
        return x[0]

    def training_step(self, batch, batch_idx):
        x, y = batch
        spike_hat = self(x)
        loss = self.loss(spike_hat, y)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        spike_hat = self(x)
        loss = self.loss(spike_hat, y)
        self.log("val_loss", loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        spike_hat = self(x)
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