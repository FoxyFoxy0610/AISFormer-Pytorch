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
├── model/
│   ├── __init__.py
│   ├── model.py               # Main model architecture (Aligned RoI, Light Mask Head)
│   ├── dataset.py             # KINS dataset loader (handles amodal/inmodal mask conversion)
│   ├── transformer.py         # Transformer Encoder/Decoder
│   ├── position_encoding.py   # Position encoding
│   └── mlp.py                 # MLP classifier/regressor
├── datasets/
│   └── KINS/                  # KINS dataset storage
│       ├── instances_train.json  # Training annotations
│       ├── instances_val.json    # Validation annotations
│       ├── training/
│       │   └── image_2/          # Training and validation images
│       └── testing/
│           └── image_2/          # Testing images
├── output/                    # Training outputs and checkpoints
├── train.py                   # Training script
├── evaluate.py                # Evaluation script
└── demo.py                    # Inference visualization script
```

## Hyperparameters and Official Setup Comparison

This framework aligns with the official hyperparameters as much as possible. However, please note the following settings during execution:

| Parameter | Current Default Setting | Official / Common Setting | Description / Recommendation |
|---|---|---|---|
| **Transformer Head** | `n_heads=2`, `dim_ff=2048` | `n_heads=2`, `dim_ff=2048` | Fully aligned. |
| **Mask Loss Weight** | Default is `1.0` | Occluder and Invisible Mask is `0.25` | **Crucial:** You must add the `--official_loss_weight` flag during training to enable the official 0.25 weight. |
| **Optimizer** | AdamW (lr=1e-4, wd=0.05) | AdamW (lr=1e-4, wd=0.05) | Aligned. |
| **Learning Rate Schedule** | `CosineAnnealingLR` | `MultiStepLR` (typically drops LR at 24k/44k steps) | Currently using Cosine Annealing for smooth decay. To replicate the official setup exactly, change this to MultiStepLR in `train.py`. |
| **Batch Size** | Default `--batch_size 1` | Typically `8` or `16` (across multiple GPUs) | Recommendation: Use `--grad_accum 8` or `--grad_accum 16` during training to simulate the official effective batch size. |

## Standard Operating Procedure (SOP)

### 1. Model Training

Ensure the KINS dataset is placed according to the directory structure above, then execute the following command to start training:

```bash
# It is recommended to enable official_loss_weight and adjust grad_accum to simulate a larger batch size
python train.py --batch_size 1 --grad_accum 8 --official_loss_weight
```

**Training Output:**
Model weights and TensorBoard logs will be saved in `output/kins_2026/run_official/` (or a custom path via `--output_dir`).

### 2. Model Evaluation

Use the validation set (`instances_val.json`) to evaluate AP (Average Precision):

```bash
python evaluate.py --checkpoint output/kins_2026/run_official/model_best.pth --official_loss_weight
```

### 3. Model Inference and Visualization (Demo)

Run inference on a single image or a batch of images and output Amodal/Visible/Invisible Mask visualizations:

```bash
python demo.py --checkpoint output/kins_2026/run_official/model_best.pth --input_dir datasets/KINS/testing/image_2 --output_dir output/demo_results
```
*(Note: If the number of classes is hardcoded to 5 in `demo.py`, please change it to 9 to match the KINS dataset.)*
