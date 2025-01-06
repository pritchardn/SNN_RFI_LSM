"""
Rate-coding spike encoder and decoder.
"""

import numpy as np
import torch
from snntorch import spikegen

from interfaces.data.spiking_data_module import SpikeConverter


class RateFullRelativeSpikeConverter(SpikeConverter):
    def __init__(self, exposure: int):
        self.exposure = exposure

    def encode_x(self, x_data: np.ndarray) -> np.ndarray:
        pre_sized = self.encode_y(x_data)[..., :x_data.shape[-2]]  # Remove duplicate
        pre_sized = np.transpose(pre_sized, (1, 0, 2))
        encoded = spikegen.rate(
            torch.from_numpy(pre_sized), time_var_input=True
        ).numpy()
        encoded = np.transpose(encoded, (1, 0, 2))
        return encoded.astype(np.float32)

    def encode_y(self, y_data: np.ndarray) -> np.ndarray:
        """
        Returns an [N x time x freq] array, where each entry in [N, T] is a -1 padded list of
        frequencies we treat as classification targets.
        This is a little confusing at first, but is done to make loss calculation easier.
        :param y_data:
        :return:
        """
        encoded = np.expand_dims(y_data, axis=1)
        encoded = np.repeat(encoded, self.exposure, axis=1)
        channel_dim = encoded.shape[-2]
        out = np.zeros(
            (encoded.shape[0], encoded.shape[-1] * self.exposure, channel_dim * 2)
        )
        for i in range(len(encoded)):
            for j in range(encoded[i].shape[-1]):
                out[i, j * self.exposure : (j + 1) * self.exposure, :channel_dim] = encoded[
                    i, ..., 0, :, j
                ]
                out[i, j * self.exposure : (j + 1) * self.exposure, channel_dim:] = 1 - encoded[
                    i, ..., 0, :, j
                ]
        return out.astype(np.float32)

    def plot_sample(
        self, x_data: np.ndarray, y_data: np.ndarray, output_dir: str, num: int
    ):
        pass

    def decode_inference(self, inference: np.ndarray) -> np.ndarray:
        # Reshape to separate the 'exp' dimension
        N, exp_T, C = inference.shape
        T = exp_T // self.exposure
        reshaped = inference.reshape(N, self.exposure, T, C)

        # Calculate the mean across the 'exp' dimension
        mean_inference = reshaped.mean(axis=1)

        # Reshape to the desired output shape (N, 1, F, T)
        output = mean_inference.transpose(0, 2, 1)[:, np.newaxis, :, :]
        output = np.maximum(output[:, :, :C//2], output[:, :, C//2:])
        return (output > 0.2).astype(np.float32)

    def decode_inference_training(self, inference: torch.Tensor):
        # Reshape to separate the 'exp' dimension
        N, exp_T, C = inference.shape
        T = exp_T // self.exposure
        reshaped = inference.reshape(N, self.exposure, T, C)

        # Calculate the mean across the 'exp' dimension
        mean_inference = reshaped.mean(axis=1)
        output = torch.maximum(mean_inference[:, :, :C//2], mean_inference[:, :, C//2:])
        return torch.sigmoid(output - 0.2)

    def decode_y(self, y_data: np.ndarray) -> np.ndarray:
        # Undo the encode_y operation
        return y_data[:, :: self.exposure, :y_data.shape[-1]//2]
