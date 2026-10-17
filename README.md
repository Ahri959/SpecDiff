# SpecDiff

**SpecDiff: Reliability-Aware Spectral Priors with One-Step Diffusion for Alignment-free RGB-T Salient Object Detection**

SpecDiff separates reliability-aware spectral structural-prior recovery from
prior-conditioned saliency completion for alignment-free RGB-T inputs.

## Release status

The model implementation is available in this repository.
**Pretrained weights and training scripts will be released upon acceptance.**

This release contains model definitions and their dependencies. Training,
inference, evaluation, and dataset-generation scripts are not included.

## Model components

- **R-ASE:** coarse thermal-to-RGB feature alignment, frequency-dependent
  uncertainty estimation, and reliability-aware spectral fusion.
- **One-step completion:** a pretrained Stable Diffusion 2.1-base VAE/UNet
  maps the RGB latent to a completed latent conditioned on the recovered prior.
- **Saliency decoder:** decodes the completed latent with UNet feature guidance.
- **CMCD:** supervises latent-prior correspondence using matched, completed,
  and sample-shuffled mismatched pairs.

## Repository structure

```text
diffusion/
  specdiff.py                  # SpecDiff model and conditioning interface
  checkpoint.py                # Model checkpoint loading
  models/
    rase.py                    # R-ASE feature extractor and structural prior
    spectral_fusion.py         # Spectral uncertainty, risk modulation and fusion
    decoder.py                 # Saliency decoder
    discriminator.py           # Cross-Modal Consistency Discriminator
    autoencoder_kl.py           # Adapted VAE
    unet_2d_condition.py        # Adapted conditional UNet
```

## Dependencies

Use Python 3.10 and install the dependencies with:

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
conda env create -f environment.yml
conda activate specdiff
```

CUDA-enabled PyTorch and matching torchvision deformable-convolution operators
are required for the complete model. Stable Diffusion 2.1-base pretrained
components must be obtained separately under the model provider's terms.

## Implementation notes

The default diffusion timestep is fixed to **499** for training and inference.
R-ASE prior features are pooled to an 8 x 8 grid and projected into 64
conditioning tokens of dimension 1024. These tokens provide the keys and
values in UNet cross-attention; no textual prompt is used.

The frequency-risk map is expanded along the feature-channel dimension to
modulate thermal uncertainty. The default discrepancy coefficient is
`lambda_u = 2`. The phase gate uses a learned positive scale and fixed bounds
of 0.05 and 0.95; it is not identical to the normalized amplitude weight.

CMCD uses shuffled mismatched-pair supervision without an additional
compatibility head or pair-ranking objective. Existing checkpoint dictionaries
retain their parameter keys. Optional decoder/quality components are restored
only when required by the loaded checkpoint.

The model takes RGB tensors of shape `[B,3,H,W]` and single-channel thermal
tensors of shape `[B,1,H,W]`, normalized to `[-1,1]`. The default input
resolution is 384 x 384. The saliency decoder returns logits; applying sigmoid
produces saliency probabilities.

## Saliency Prediction Maps

To facilitate reproducibility, independent verification, and fair comparison, we provide the predicted saliency maps of **all evaluated methods** used in our experiments.

The complete collection of prediction maps is publicly available:

📁 **[Download Saliency Prediction Maps (Google Drive)](https://drive.google.com/drive/folders/13WTnasPLoVO9lE71RQjdQM4FPZLmH_2F?usp=drive_link)**

These resources are provided to support further research and facilitate comparisons with existing methods.
