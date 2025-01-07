import numpy as np
import torch

from interfaces.data.spiking_data_module import SpikeConverter


class DirectMembraneConverter(SpikeConverter):
    def __init__(self, exposure: int):
        self.exposure = exposure

    def encode_x(self, x_data: np.ndarray) -> np.ndarray:
        pre_sized = self.encode_y(x_data)
        return pre_sized.astype(np.float32)

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
        out = np.zeros(
            (encoded.shape[0], encoded.shape[-1] * self.exposure, encoded.shape[-2])
        )
        for i in range(len(encoded)):
            for j in range(encoded[i].shape[-1]):
                out[i, j * self.exposure: (j + 1) * self.exposure, :] = encoded[
                                                                        i, ..., 0, :, j
                                                                        ]
        return out.astype(np.float32)

    def plot_sample(
            self, x_data: np.ndarray, y_data: np.ndarray, output_dir: str, num: int
    ):
        pass

    def decode_inference(self, inference: np.ndarray) -> np.ndarray:
        return inference.astype(np.float32)
        # Inference is now a membrane potential, not series of spikes
        # Reshape to separate the 'exp' dimension
        N, exp_T, C = inference.shape
        T = exp_T // self.exposure
        reshaped = inference.reshape(N, self.exposure, T, C)

        # Collect the final membrane potential for each self.exposure period
        final_membrane = reshaped[:, ::self.exposure, :, :]

        # Reshape to the desired output shape (N, 1, F, T)
        return final_membrane.astype(np.float32)

    def decode_inference_training(self, inference: torch.Tensor):
        # Reshape to separate the 'exp' dimension
        return inference
        N, exp_T, C = inference.shape
        T = exp_T // self.exposure
        reshaped = inference.reshape(N, self.exposure, T, C)

        # Collect the final membrane potential for each self.exposure period
        final_membrane = reshaped[:, ::self.exposure, :, :]
        return torch.sigmoid(final_membrane)

    def decode_y(self, y_data: np.ndarray) -> np.ndarray:
        # Undo the encode_y operation
        return y_data[:, :: self.exposure, :]