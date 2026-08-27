# AISFormer-Pytorch

[Official GitHub (UARK-AICV/AISFormer)](https://github.com/UARK-AICV/AISFormer) | [Paper (arXiv:2210.06323)](https://arxiv.org/abs/2210.06323)

## Project Background and Motivation

This project is an independently developed, pure PyTorch implementation based on the Amodal Instance Segmentation (AIS) architecture proposed by the [UARK-AICV/AISFormer](https://github.com/UARK-AICV/AISFormer) team.

The open-source environment provided by the original authors is built on a customized and modified Detectron2 framework. However, this customized environment often faces forward compatibility challenges when deployed on modern server environments, making it difficult to install and run directly. Furthermore, the original framework is overly restrictive regarding the selection and replacement of the feature extraction backbone, making it difficult to adapt flexibly to different hardware and project requirements.

To address these pain points, this project re-develops the same model architecture purely in PyTorch from an engineering application perspective. By removing the dependency on Detectron2, we have restored the model performance and algorithmic details from the original paper as much as possible, providing a cleaner framework that is more flexible, easier to maintain, and readily allows for feature backbone expansion.

---

## Directory Structure

```text
AISFormer-Pytorch/
?œâ??€ model/
??  ?œâ??€ __init__.py
??  ?œâ??€ model.py               # Main model architecture (Aligned RoI, Light Mask Head)
??  ?œâ??€ dataset.py             # KINS dataset loader (handles amodal/inmodal mask conversion)
??  ?œâ??€ transformer.py         # Transformer Encoder/Decoder
??  ?œâ??€ position_encoding.py   # Position encoding
??  ?”â??€ mlp.py                 # MLP classifier/regressor
?œâ??€ datasets/
??  ?”â??€ KINS/                  # KINS dataset storage
??      ?œâ??€ instances_train.json  # Training annotations
??      ?œâ??€ instances_val.json    # Validation annotations
??      ?œâ??€ training/
??      ??  ?”â??€ image_2/          # Training and validation images
??      ?”â??€ testing/
??          ?”â??€ image_2/          # Testing images
?œâ??€ output/                    # Training outputs and checkpoints
?œâ??€ train.py                   # Training script
?œâ??€ evaluate.py                # Evaluation script
?œâ??€ demo.py                    # Inference visualization script
?”â??€ requirements.txt           # Environment dependencies
```

## Hyperparameters and Official Setup Comparison

This framework aligns with the official hyperparameters as much as possible. However, please note the following settings during execution:

| Parameter | Current Default Setting | Official / Common Setting | Description / Recommendation |
|---|---|---|---|
| **Transformer Head** | `n_heads=2`, `dim_ff=2048` | `n_heads=2`, `dim_ff=2048` | Fully aligned. |
| **Mask Loss Weight** | Default is `1.0` | Occluder and Invisible Mask is `0.25` | **Crucial:** You must add the `--official_loss_weight` flag during training to enable the official 0.25 weight. |
| **Optimizer & LR** | AdamW (lr=1e-4, wd=0.05) | SGD (lr=0.0025) | The official literature uses SGD, but AdamW is provided here as the default for better stability. |
| **Learning Rate Schedule** | `CosineAnnealingLR` (50,000 iters) | `MultiStepLR` (48,000 iters, drops at 24k/44k) | Currently using Cosine Annealing over 50,000 iterations for smooth decay. To replicate the official setup exactly, change this to MultiStepLR in `train.py`. |
| **Batch Size** | Default `--batch_size 1` | Typically `8` or `16` (across multiple GPUs) | Recommendation: Use `--grad_accum 8` or `--grad_accum 16` during training to simulate the official effective batch size. |


## Backbone Selection & Engineering Insights

Extensive experiments were conducted during the development of this PyTorch version. The following insights regarding backbone selection and feature extraction are shared to help adapt this framework to various downstream tasks:

*   **Resolution and Domain Shift:** Low resolution feature extraction helps improve robustness under domain shift, which is highly beneficial when using architectures like **Swin** and **ConvNeXt**.
*   **Curse of Dimensionality:** High-resolution CNN feature extraction can easily cause the model to fall into local minima.
*   **ConvNeXt v2:** The introduction of **GRN** (Global Response Normalization) in ConvNeXt v2 effectively eases overfitting issues. However, it must be paired with suitable dropout and stochastic depth methods to maximize its potential.
*   **Swin Transformer v2 Limitation:** **Swin Transformer v2** is not recommended as a backbone for small datasets. Its **scaled cosine attention** mechanism is designed for large-scale dataset training; applying it to smaller datasets often results in stagnant loss curves.
*   **Swin for Specific Contexts:** Swin-based backbones are particularly suitable for detecting objects with complex, repeating patterns (e.g., flowers or specific textures) due to their Hierarchical and Shifted Window Attention.
*   **Overfitting Mitigation:** When using the Swin Transformer on small datasets (where feature differences are minimal), aggressive augmentations like **Copy-Paste** and severe color distortion are highly effective at fixing overfitting issues.
*   **Dropout and Stochastic Depth:** In standard experimental settings, the **Dropout rate** and Stochastic Depth are often defined as 0.5 across all backbone versions for consistency. However, literature on ConvNeXt indicates that smaller backbone variants only require a rate of 0.1 for sufficient regularization. Setting the Dropout rate too high on these smaller models may cause underfitting.
*   **Ongoing Research:** The exact reasons behind the performance discrepancies across various backbone types and scaling strategies in amodal segmentation still require further clarification.

## Command-Line Arguments

The training script provides several arguments for flexible configuration. A key advantage of this PyTorch implementation over the official Detectron2 version is the **highly extensible backbone support**. You are no longer restricted to a few predefined backbones; you can easily plug in almost any modern architecture (e.g., ConvNeXt, Swin Transformer, RegNet) via `timm` and `torchvision`.

*   `--backbone`: Specifies the feature extraction backbone (default: `resnet50`). This framework natively supports a massive array of PyTorch and `timm` backbones. Commonly supported variants include:
    *   **ResNet Series**: `resnet50`, `resnet101`
    *   **Swin Transformer**: `swin_t`, `swin_s`, `swin_b`
    *   **ConvNeXt (v1/v2)**: `convnext_tiny`, `convnext_small`, `convnext_base`, `convnextv2_tiny`, `convnextv2_base`
    *   **RegNet Series**: `regnet_y_3_2gf`, `regnet_y_8gf`, `regnet_y_16gf`
*   `--batch_size`: Physical batch size per GPU (default: `1`).
*   `--grad_accum`: Number of gradient accumulation steps (default: `1`). Use this to simulate larger batch sizes on hardware with limited VRAM.
*   `--official_loss_weight`: Applies a `0.25` multiplier to the occluder and invisible mask losses, replicating the exact loss weighting from the original paper.
*   `--lr`: Learning rate for the AdamW optimizer (default: `1e-4`).
*   `--output_dir`: Directory to save model checkpoints and TensorBoard logs.
*   `--resume`: Path to a specific checkpoint file to resume training from a previous state.
*   `--use_copy_paste`: Enables Copy-Paste data augmentation (if implemented in the dataset pipeline).

## Standard Operating Procedure (SOP)

### 1. Environment Setup

Install the required dependencies via the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 2. Model Training

Ensure the KINS dataset is placed according to the directory structure above, then execute the following command to start training:

```bash
# It is recommended to enable official_loss_weight and adjust grad_accum to simulate a larger batch size
python train.py --batch_size 1 --grad_accum 8 --official_loss_weight
```

**Training Output:**
Model weights and TensorBoard logs will be saved in `output/kins_2026/run_official/` (or a custom path via `--output_dir`).

### 3. Model Evaluation

Use the validation set (`instances_val.json`) to evaluate AP (Average Precision):

```bash
python evaluate.py --checkpoint output/kins_2026/run_official/model_best.pth --official_loss_weight
```

### 4. Model Inference and Visualization (Demo)

Run inference on a single image or a batch of images and output Amodal/Visible/Invisible Mask visualizations:

```bash
python demo.py --checkpoint output/kins_2026/run_official/model_best.pth --input_dir datasets/KINS/testing/image_2 --output_dir output/demo_results
```
*(Note: If the number of classes is hardcoded to 5 in `demo.py`, please change it to 9 to match the KINS dataset.)*

