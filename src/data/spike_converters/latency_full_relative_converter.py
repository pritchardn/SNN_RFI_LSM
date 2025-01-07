import numpy as np
import torch
from snntorch import spikegen

from interfaces.data.spiking_data_module import SpikeConverter


class LatencyFullRelativeSpikeConverter(SpikeConverter):


    def __init__(self, exposure: int, tau: float, normalize: bool):
        self.exposure = exposure
        self.tau = tau
        self.normalize = normalize

    def encode_x(self, x_data: np.ndarray) -> np.ndarray:
        output = self.encode_y(x_data)[..., :x_data.shape[-2]]  # Remove duplicate
        return output.astype("float32")

    def encode_y(self, y_data: np.ndarray) -> np.ndarray:
        channel_dim = y_data.shape[-2]
        output_shape = (
            y_data.shape[0],
            self.exposure * y_data.shape[-1],
            channel_dim * 2,
        )
        output_timings = np.zeros(output_shape, dtype=y_data.dtype)
        for i, frame in enumerate(y_data):
            frame = torch.from_numpy(np.moveaxis(frame, 0, -1))
            frame = spikegen.latency(
                frame, num_steps=self.exposure, tau=self.tau, normalize=True
            )
            frame = np.squeeze(frame.numpy(), -1)
            for j in range(frame.shape[-1]):
                output_timings[i, j * self.exposure:(j + 1) * self.exposure, :channel_dim] = frame[
                    :, :, j
                ]

                output_timings[i, j * self.exposure:(j + 1) * self.exposure, channel_dim:] = 1 - frame[
                    :, :, j
                ]
        return output_timings

    def plot_sample(
        self, x_data: np.ndarray, y_data: np.ndarray, output_dir: str, num: int
    ):
        pass

    def decode_inference(self, inference: np.ndarray) -> np.ndarray:
        """
        Rebuilds mask from spiking output. In this case, any pixel location that did not spike until
        the end of the exposure time is considered the background.
        Assumes a shape of [N, exposure, C, freq, time]
        :return: [N, C, freq, time]
        """
        top_half = inference[:, :, :inference.shape[-1] // 2]
        bottom_half = inference[:, :, inference.shape[-1] // 2:]
        output = np.where(top_half > bottom_half, top_half, np.zeros_like(top_half))
        reshaped = output.reshape(output.shape[0], self.exposure, output.shape[-1], output.shape[-1])
        return np.expand_dims(reshaped[:, :-1, :, :].sum(axis=1), axis=1)

    def decode_inference_training(self, inference: torch.Tensor) -> torch.Tensor:
        top_half = inference[:, :, :inference.shape[-1] // 2]
        bottom_half = inference[:, :, inference.shape[-1] // 2:]
        output = torch.where(top_half > bottom_half, top_half, torch.zeros_like(top_half))
        reshaped = output.reshape(output.shape[0], self.exposure, output.shape[-1],
                                     output.shape[-1])
        return reshaped[:, :-1, :, :].sum(dim=1)

    def decode_y(self, y_data: np.ndarray) -> np.ndarray:
        if isinstance(y_data, torch.Tensor):
            return self.decode_inference_training(y_data)
        else:
            return self.decode_inference(y_data)

