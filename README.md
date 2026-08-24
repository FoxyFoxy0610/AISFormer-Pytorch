# AISFormer-Pytorch

[Official GitHub (UARK-AICV/AISFormer)](https://github.com/UARK-AICV/AISFormer) | [Paper (arXiv:2210.06323)](https://arxiv.org/abs/2210.06323)

## 專案背景與動機

本專案是基於 [UARK-AICV/AISFormer](https://github.com/UARK-AICV/AISFormer) 團隊提出的 Amodal Instance Segmentation (AIS) 架構所獨立開發的純 PyTorch 版本。

原作者提供的開源環境是基於客製化修改的 Detectron2 框架建構的。然而，該客製化環境在面對當今較新的伺服器環境時，往往會面臨**向上兼容性 (Forward Compatibility)** 的挑戰，難以直接安裝與運行。此外，原框架在**特徵擷取骨幹 (Feature Backbone)** 的選擇與替換上過於受限，難以靈活套用於不同的硬體與專案需求。

為了解決這些痛點，本專案**以工程應用的角度出發，重新以純 PyTorch 開發了相同的模型架構**。我們在脫離 Detectron2 的依賴下，盡可能地還原了原論文中的模型表現與演算法細節，提供了一個更具彈性、更好維護且能輕易擴展特徵骨幹的乾淨框架。

---

## 目錄結構

```text
AISFormer-Pytorch/
├── 📁 model/
│   ├── __init__.py
│   ├── model.py               # 主模型架構 (Aligned RoI, Light Mask Head)
│   ├── dataset.py             # KINS 資料集載入器 (負責 amodal/inmodal mask 轉換)
│   ├── transformer.py         # Transformer Encoder/Decoder
│   ├── position_encoding.py   # 位置編碼
│   └── mlp.py                 # MLP 分類器/迴歸器
├── 📁 datasets/
│   └── KINS/                  # KINS 資料集存放區
│       ├── instances_train.json  # 訓練標註
│       ├── instances_val.json    # 驗證標註
│       ├── training/
│       │   └── image_2/          # 訓練與驗證圖片
│       └── testing/
│           └── image_2/          # 測試圖片
├── 📁 output/                 # 訓練輸出與 Checkpoints
├── train.py                   # 訓練腳本
├── evaluate.py                # 評估腳本
└── demo.py                    # 推論視覺化腳本
```

## 超參數與官方設定對比

本框架已盡可能對齊官方的超參數，但在執行時需要注意以下設定：

| 參數 | 本框架目前預設設定 | 官方推薦 / 常見設定 | 說明 / 建議 |
|---|---|---|---|
| **Transformer Head** | `n_heads=2`, `dim_ff=2048` | `n_heads=2`, `dim_ff=2048` | ✅ 完全對齊 |
| **Mask 損失權重** | 預設為 `1.0` | 遮蔽物與不可見 Mask 為 `0.25` | ⚠️ 訓練時**必須**加上 `--official_loss_weight` 參數來啟用官方的 0.25 權重 |
| **優化器 (Optimizer)** | AdamW (lr=1e-4, wd=0.05) | AdamW (lr=1e-4, wd=0.05) | ✅ 對齊 |
| **學習率排程** | `CosineAnnealingLR` | `MultiStepLR` (通常於 24k/44k 步降 LR) | ℹ️ 目前採用 Cosine 退火 (平滑下降)。若要完全複製官方，可於 `train.py` 中改為 MultiStepLR |
| **Batch Size** | 預設 `--batch_size 1` | 通常為 `8` 或 `16` (跨多 GPU) | ⚠️ 建議訓練時使用 `--grad_accum 8` 或 `--grad_accum 16` 來模擬官方的 effective batch size |

## 使用 SOP

### 1. 模型訓練

請確保 KINS 資料集已按照上述目錄結構放置，然後執行以下指令開始訓練：

```bash
# 建議啟用 official_loss_weight 並調整 grad_accum 來模擬大 Batch Size
python train.py --batch_size 1 --grad_accum 8 --official_loss_weight
```

**訓練輸出：**
模型權重與 TensorBoard 日誌將會保存在 `output/kins_2026/run_official/` (或透過 `--output_dir` 自定義路徑)。

### 2. 模型評估

使用驗證集 (`instances_val.json`) 來評估 AP (Average Precision)：

```bash
python evaluate.py --checkpoint output/kins_2026/run_official/model_best.pth --official_loss_weight
```

### 3. 模型推論與視覺化 (Demo)

進行單張或多張圖片的推論，並輸出 Amodal/Visible/Invisible 的 Mask 視覺化結果：

```bash
python demo.py --checkpoint output/kins_2026/run_official/model_best.pth --input_dir datasets/KINS/testing/image_2 --output_dir output/demo_results
```
*(註：若 `demo.py` 中有寫死類別數為 5，請記得將其改為 9 以匹配 KINS 資料集)*
