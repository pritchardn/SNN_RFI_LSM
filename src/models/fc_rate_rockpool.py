import numpy as np
import torch
from torch.nn import MSELoss

from interfaces.models.model_rockpool import LitModelRockpool
from plotting import (
    plot_example_inference,
    plot_example_mask,
    plot_network_internals,
    plot_target_raster,
    plot_input_raster,
)


class LitFcRateRockpool(LitModelRockpool):
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
        self.loss = MSELoss()
        self.float()
        self.save_hyperparameters()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        spike_hat, recording = self(x)
        loss = self.calc_loss(spike_hat, y)
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
