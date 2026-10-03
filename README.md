# SNN_RFI_LSM
Liquid State Machine Approach to RFI detection with SNNs. Built with Rockpool

## Readout-only ablation
`READOUT_*` model types remove the spiking reservoir and input projection and train only the readout stack
(`Linear(num_inputs -> num_outputs)` followed by the linear / ReLU / transformer decoder) directly on the encoded
spikes. Encoder, exposure, loss, optimiser and metrics are taken from the matching `LSM*` config, so results are
directly comparable.

| Reservoir model | Readout-only counterpart |
|---|---|
| `LSM` | `READOUT_LINEAR` |
| `LSM_RELU` | `READOUT_RELU` |
| `LSM_TRANSFORMER` | `READOUT_TRANSFORMER` |
| `LSM_REL` / `LSM_RELU_REL` | `READOUT_LINEAR_REL` / `READOUT_RELU_REL` / `READOUT_TRANSFORMER_REL` |

Run from `src/` with the same environment variables as `main.py`, e.g.
`MODEL_TYPE=READOUT_RELU ENCODER_METHOD=RATE_FULL DATA_PATH=./data python main.py`.
