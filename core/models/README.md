# `face2eeg/core/models/README.md`

# `face2eeg.core.models` — Networks, losses, and checkpointing

This subpackage contains everything needed to build, train, save, and
reload the neural network used to predict EEG from facial video. It has
no dependency on the data layer — models operate purely on tensors — and
only a light dependency on `torch`.

Three modules are exposed:

| Module        | Responsibility                                        | Torch-free? |
|---------------|-------------------------------------------------------|:-----------:|
| `physnet.py`  | `PhysNet_3DCNN_ED` encoder–decoder                    | No          |
| `losses.py`   | `Neg_Pearson` correlation-based loss                  | No          |
| `checkpoint.py` | device selection and model reload helpers           | No          |

---

## Table of contents

- [Network: PhysNet-3DCNN-ED](#network-physnet-3dcnn-ed)
  - [Reference implementation](#reference-implementation)
  - [Architecture overview](#architecture-overview)
  - [Layer-by-layer](#layer-by-layer)
  - [Input / output shapes](#input--output-shapes)
  - [Design notes vs the original PhysNet](#design-notes-vs-the-original-physnet)
- [Losses](#losses)
- [Checkpoint utilities](#checkpoint-utilities)
- [Typical usage](#typical-usage)
- [Related documentation](#related-documentation)

---

## Network: PhysNet-3DCNN-ED

### Reference implementation

The architecture is a direct adaptation of the **PhysNet** 3D-CNN
encoder–decoder originally proposed for remote photoplethysmography
(rPPG) from facial video:

> **Yu, Z., Li, X., & Zhao, G. (2019).** *Remote Photoplethysmograph
> Signal Measurement from Facial Videos Using Spatio-Temporal Networks.*
> In Proceedings of the 28th International Joint Conference on
> Artificial Intelligence (IJCAI), pp. 2857–2864.
>
> Original code: **https://github.com/ZitongYu/PhysNet/tree/master**

The original PhysNet predicts a 1-D blood-volume-pulse waveform from a
sequence of facial frames. In this project we reuse the exact
encoder–decoder skeleton, with only the output length adjusted to match
the target EEG segment length.

### Architecture overview

`PhysNet_3DCNN_ED` is a 3D convolutional encoder–decoder with no fully
connected layers and no temporal pooling in the head:

- **Input:** `(B, 3, T, W, H)` — a batch of RGB video segments.
- **Encoder:** 9 3D convolution blocks, all with BatchNorm and ReLU,
  interspersed with max-pooling that halves the spatial and, at two
  points, the temporal axes.
- **Decoder:** two transposed 3D convolutions that restore the original
  temporal length `T`.
- **Head:** adaptive average pooling over the spatial dimensions
  followed by a 1×1×1 convolution and a reshape to `(B, signal_length)`.
- **Output:** `(B, signal_length)` — a 1-D EEG waveform per segment.

The design is fully convolutional along the time axis, so any temporal
length `T` is accepted at inference, provided the internal pooling
hierarchy divides it evenly.

### Layer-by-layer

| Block          | Operation                                              | Out channels | Spatial | Temporal |
|----------------|--------------------------------------------------------|:------------:|:-------:|:--------:|
| `ConvBlock1`   | `Conv3d(3 → 16, k=(1,5,5), p=(0,2,2))` + BN + ReLU     | 16           | W, H    | T        |
| `MaxpoolSpa`   | `MaxPool3d((1,2,2), stride=(1,2,2))`                   | 16           | W/2, H/2| T        |
| `ConvBlock2`   | `Conv3d(16 → 32, k=3, p=1)` + BN + ReLU                | 32           | W/2, H/2| T        |
| `ConvBlock3`   | `Conv3d(32 → 64, k=3, p=1)` + BN + ReLU                | 64           | W/2, H/2| T        |
| `MaxpoolSpaTem`| `MaxPool3d((2,2,2), stride=2)`                         | 64           | W/4, H/4| T/2      |
| `ConvBlock4`   | `Conv3d(64 → 64, k=3, p=1)` + BN + ReLU                | 64           | W/4, H/4| T/2      |
| `ConvBlock5`   | `Conv3d(64 → 64, k=3, p=1)` + BN + ReLU                | 64           | W/4, H/4| T/2      |
| `MaxpoolSpaTem`| `MaxPool3d((2,2,2), stride=2)`                         | 64           | W/8, H/8| T/4      |
| `ConvBlock6`   | `Conv3d(64 → 64, k=3, p=1)` + BN + ReLU                | 64           | W/8, H/8| T/4      |
| `ConvBlock7`   | `Conv3d(64 → 64, k=3, p=1)` + BN + ReLU                | 64           | W/8, H/8| T/4      |
| `MaxpoolSpa`   | `MaxPool3d((1,2,2), stride=(1,2,2))`                   | 64           | W/16,H/16| T/4     |
| `ConvBlock8`   | `Conv3d(64 → 64, k=3, p=1)` + BN + ReLU                | 64           | W/16,H/16| T/4     |
| `ConvBlock9`   | `Conv3d(64 → 64, k=3, p=1)` + BN + ReLU                | 64           | W/16,H/16| T/4     |
| `upsample`     | `ConvTranspose3d(64→64, k=(4,1,1), s=(2,1,1), p=(1,0,0))` + BN + ELU | 64 | W/16,H/16 | T/2 |
| `upsample2`    | `ConvTranspose3d(64→64, k=(4,1,1), s=(2,1,1), p=(1,0,0))` + BN + ELU | 64 | W/16,H/16 | T  |
| `poolspa`      | `AdaptiveAvgPool3d((signal_length, 1, 1))`             | 64           | 1, 1    | signal_length |
| `ConvBlock10`  | `Conv3d(64 → 1, k=1)`                                  | 1            | 1, 1    | signal_length |
| `view`         | reshape to `(B, signal_length)`                        | 1            | –       | signal_length |

### Input / output shapes

| Axis         | Symbol            | Typical value in this project |
|--------------|-------------------|--------------------------------|
| Batch        | `B`               | 16 (training) / 16 (evaluation) |
| Channels     | `C`               | 3 (RGB)                        |
| Time         | `T`               | 50 (2 s @ 25 fps facial video) or 100 (T=2 s @ 50 fps) |
| Width        | `W`               | 32                             |
| Height       | `H`               | 32                             |
| **Output**   | `signal_length`   | `T_signal = time_window × 128 Hz` (e.g. 256 for 2 s) |

The model is fully convolutional, so the same instance accepts any
segment length at inference as long as it was trained on a compatible
range. In this project, `video_length` and `signal_length` are set at
construction and stored in the checkpoint.

### Design notes vs the original PhysNet

| Aspect                       | Original PhysNet (Yu et al.)  | `PhysNet_3DCNN_ED` (this project)             |
|------------------------------|-------------------------------|-----------------------------------------------|
| Target signal                | 1-D rPPG waveform             | 1-D EEG waveform                              |
| Output length                | ~ 128–256 samples             | `signal_length = time_window × 128`           |
| Temporal pooling             | Same scheme                   | Same scheme                                   |
| Spatial pooling              | Same scheme                   | Same scheme                                   |
| Final head                   | `AdaptiveAvgPool3d` + 1×1 conv| Same                                          |
| Number of parameters         | ≈ 1.8 M                       | Same (identical layer stack)                  |
| Loss function used in training | Neg-Pearson + MSE variants  | MSE (main), Neg-Pearson (comparison)          |

The only semantic change is the meaning of the output: PhysNet predicts
a blood-volume pulse, `PhysNet_3DCNN_ED` predicts EEG. The mapping is
learned end-to-end from facial video.

**Why we did not modify the architecture**: the reviewer's and the
original paper's motivation for the 3D-CNN encoder–decoder is that it
preserves temporal structure through the bottleneck, and the two
upsampling layers on the time axis restore the target sampling rate.
Any architectural change would have required re-tuning and would have
made the comparison against the original PhysNet baseline inexpressive.

**A note on the `signal_length` argument**: it controls only the
adaptive pooling target and the final reshape. All convolutional
weights are independent of it, so the same trained weights can be
applied to inputs of different temporal lengths *only if*
`AdaptiveAvgPool3d` is re-set — in practice we retrain for each
`signal_length`.

---

## Losses

### `Neg_Pearson` — negative Pearson correlation

```python
class Neg_Pearson(nn.Module):
    def forward(self, preds, labels):
        ...
        return loss   # scalar
```

Computes the Pearson correlation between the predicted and target
waveforms **per batch item** and averages `1 − r` across the batch.
The loss is minimized when `r = 1` for every item.

**Range and gradient behaviour:**

- `r ∈ [−1, 1]`, so `loss ∈ [0, 2]`.
- Minimizing `1 − r` directly maximizes correlation; the model is
  rewarded for tracking temporal shape, not amplitude.
- Because the loss is scale- and offset-invariant, it must be combined
  with z-scoring of the targets during training (which is what
  `signal_normalize=True` does in the data pipeline).

**Relation to MSE:** in the paper, MSE is used as the primary loss and
`Neg_Pearson` as a comparison (Experiment 1). The MSE-trained model
achieves a higher correlation in practice because it penalizes
amplitude errors as well as shape mismatches.

---

## Checkpoint utilities

### `try_gpu`

```python
def try_gpu(i=0):
    """Return cuda(i) if available, otherwise cpu()."""
```

Small helper that centralizes the device-selection logic used at the
top of every training notebook.

### `load_model`

```python
def load_model(video_length, signal_length, model_path, device):
    model = PhysNet_3DCNN_ED(video_length, signal_length)
    model = model.to(device=device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
```

Rebuilds a `PhysNet_3DCNN_ED` with the correct I/O shapes and loads the
weights from a checkpoint written by `train_model`. Raises
`ModuleNotFoundError` if the checkpoint file does not exist — this is
intentional, so that a caller who expects a model never silently gets
a randomly initialized one.

The checkpoint format is defined in `face2eeg.core.training`:

| Key                    | Description                                     |
|------------------------|-------------------------------------------------|
| `epoch`                | Last completed epoch                            |
| `model_state_dict`     | Model weights                                   |
| `optimizer_state_dict` | Optimizer state (for resume)                    |
| `scheduler_state_dict` | LR scheduler state (for resume)                 |
| `loss`                 | Training loss per epoch                         |
| `val_loss`             | Validation loss per epoch                       |
| `lr`                   | LR per epoch                                    |
| `best_val`             | Best validation loss seen so far                |
| `best_epoch`           | 1-based epoch of `best_val`                     |

---

## Typical usage

### Train one fold from scratch

```python
from face2eeg.core.models   import PhysNet_3DCNN_ED, Neg_Pearson, try_gpu
from face2eeg.core.training import train_model

device = try_gpu()
model  = PhysNet_3DCNN_ED(video_length=100, signal_length=256).to(device)

history = train_model(
    model, X_train, y_train,
    device=device,
    epochs=10,
    batch_size=16,
    learning_rate=1e-4,
    weight_decay=1e-5,
    loss_fn=Neg_Pearson(),            # or nn.MSELoss()
    model_save_path='fold0.pt',
)
```

### Reload from checkpoint

```python
from face2eeg.core.models import load_model, try_gpu

device = try_gpu()
model  = load_model(video_length=100, signal_length=256,
                    model_path='fold0.pt', device=device)
model.eval()
```

### Compare MSE and Neg-Pearson

```python
import torch.nn as nn
from face2eeg.core.models import Neg_Pearson

for name, criterion in [('MSE', nn.MSELoss()), ('NegPearson', Neg_Pearson())]:
    model = PhysNet_3DCNN_ED(100, 256).to(device)
    train_model(model, X_train, y_train, device=device,
                epochs=20, loss_fn=criterion,
                model_save_path=f'model_{name}.pt')
```

The trained models produce identical output shapes; only the
optimization signal differs. This is exactly the comparison reported
in Experiment 1 of the paper.

---

## Related documentation

- [`../../README.md`](../../README.md) — project overview
- [`../README.md`](../README.md) — core library
- [`../experiments/README.md`](../experiments/README.md) — end-to-end pipelines
- [`../training.py`](../training.py) — training loop and checkpoint format
- [`../data/README.md`](../data/README.md) — how video segments reach the network
- **Original PhysNet:** https://github.com/ZitongYu/PhysNet/tree/master