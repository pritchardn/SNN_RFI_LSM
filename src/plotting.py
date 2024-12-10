"""
This module contains functions for plotting the results of the inference process.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import snntorch.spikeplot as spl
import torch
from rockpool import TSEvent


def plot_example_inference(example: torch.tensor, name: str, log_dir: str):
    fig, ax = plt.subplots()
    anim = spl.animator(example, fig, ax, interval=100)
    anim.save(os.path.join(log_dir, f"example_{name}.gif"))
    plt.close(fig)


def plot_example_mask(mask: np.ndarray, name: str, log_dir: str):
    fig, ax = plt.subplots()
    ax.imshow(mask)
    plt.savefig(os.path.join(log_dir, f"example_mask_{name}.png"))
    plt.close(fig)


def plot_image_patch(
    image: np.ndarray, filename_prefix: str, output_dir: str, cbar=False
):
    plt.figure(figsize=(5, 5))
    plt.imshow(image, vmin=0, vmax=1, aspect="equal", interpolation="nearest")
    plt.ylabel("Frequency Bins")
    plt.xlabel("Time [s]")
    if cbar:
        plt.colorbar(location="right", shrink=0.8)
    plt.gca().invert_yaxis()
    plt.savefig(
        os.path.join(output_dir, f"{filename_prefix}_image.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")


def plot_final_examples(
    x_orig: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, name: str, log_dir: str
):
    # plot original
    plot_image_patch(x_orig, f"{name}_image", log_dir, cbar=True)
    # plot decoded mask
    plot_image_patch(y_pred, f"{name}_inference", log_dir)
    # plot real mask
    plot_image_patch(y_true, f"{name}_mask", log_dir)


def plot_layer_events(recording, layer_name, epoch: int, log_dir: str):
    data = recording[layer_name][0].detach().cpu().T
    plt.imshow(data, aspect="auto", origin="lower")
    plt.title("Layer: " + layer_name + " Events")
    plt.ylabel("Channel")
    plt.xlabel("Time")
    plt.savefig(os.path.join(log_dir, f"layer_{epoch}_{layer_name}_events.png"))
    plt.clf()


def plot_layer_raster(recording, layer_name, epoch: int, log_dir: str):
    data = recording[layer_name][0].detach().cpu()
    TSEvent.from_raster(data).plot()
    plt.title("Layer: " + layer_name + " Raster")
    plt.ylabel("Channel")
    plt.xlabel("Time")
    plt.savefig(os.path.join(log_dir, f"layer_{epoch}_{layer_name}_raster.png"))
    plt.clf()


def plot_target_raster(target, epoch: int, log_dir: str):
    TSEvent.from_raster(target).plot()
    plt.title("Target Raster")
    plt.ylabel("Channel")
    plt.xlabel("Time")
    plt.savefig(os.path.join(log_dir, f"target_{epoch}_raster.png"))
    plt.clf()


def plot_input_raster(input, epoch: int, log_dir: str):
    TSEvent.from_raster(input).plot()
    plt.title("Input Raster")
    plt.ylabel("Channel")
    plt.xlabel("Time")
    plt.savefig(os.path.join(log_dir, f"input_{epoch}_raster.png"))
    plt.clf()


def plot_taus(layer, epoch: int, log_dir: str):
    layer_name = str.replace(layer.name, "'", "")
    data = layer.tau_mem.detach().cpu()
    if data.dim() != 0:
        plt.hist(data * 1e3, 20)
        plt.title("Layer: " + layer_name + " Tau Mem")
        plt.ylabel("Frequency")
        plt.xlabel("Tau Mem [ms]")
        plt.savefig(os.path.join(log_dir, f"layer_{epoch}_{layer_name}_tau_mem.png"))
        plt.clf()
    data = layer.tau_syn.detach().cpu()
    if data.dim() != 0:
        plt.hist(data * 1e3, 20)
        plt.ylabel("Frequency")
        plt.xlabel("Tau Syn [ms]")
        plt.title("Layer: " + layer_name + " Tau Mem")
        plt.savefig(os.path.join(log_dir, f"layer_{epoch}_{layer_name}_tau_mem.png"))
        plt.clf()


def plot_weight_distribution(layer, epoch: int, log_dir: str):
    layer_name = str.replace(layer.name, "'", "")
    data = np.ravel(layer.weight.detach().cpu().numpy())
    plt.hist(data, 20)
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.title("Layer: " + layer_name + " Weight Distribution")
    plt.savefig(
        os.path.join(log_dir, f"layer_{epoch}_{layer_name}_weight_distribution.png")
    )
    plt.clf()


def plot_weights(layer, epoch: int, log_dir: str):
    layer_name = str.replace(layer.name, "'", "")
    data = layer.weight.detach().cpu().numpy().T
    plt.imshow(data, aspect="auto", origin="lower")
    plt.title("Layer: " + layer_name + " Weights")
    plt.ylabel("Output Channel")
    plt.xlabel("Input Channel")
    plt.colorbar()
    plt.savefig(os.path.join(log_dir, f"layer_{epoch}_{layer_name}_weights.png"))
    plt.clf()


def plot_bias(layer, epoch: int, log_dir: str):
    layer_name = str.replace(layer.name, "'", "")
    data = layer.bias.detach().cpu()
    if data.dim() != 0:
        plt.hist(data, 20)
        plt.title("Layer: " + layer_name + " Bias")
        plt.xlabel("Bias Current")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(log_dir, f"layer_{epoch}_{layer_name}_bias.png"))
        plt.clf()


def plot_network_internals(network, recording, epoch: int, log_dir: str):
    for i, layer in enumerate(network):
        if "LIF" in layer.name:
            layer_name = str.replace(layer.name, "'", "")
            plot_layer_events(recording, layer_name + "_output", epoch, log_dir)
            plot_layer_raster(recording, layer_name + "_output", epoch, log_dir)
            plot_taus(network[i], epoch, log_dir)
            plot_bias(network[i], epoch, log_dir)
        if "Linear" in layer.name:
            plot_weight_distribution(network[i], epoch, log_dir)
            plot_weights(network[i], epoch, log_dir)
