# GS-ASM: 2DGS-Supervised Active Stereo Matching

> Deep active stereo network training with synthesized proxy labels and hybrid supervision regularization.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Training](#training)
- [Testing](#testing)

## 1. Introduction

We propose a novel framework that synthesizes proxy labels to enable supervised training of deep active stereo networks without requiring any ground-truth depth. To expand the training data and generate disparity proxy labels, we develop an active 2D Gaussian Splatting (2DGS)-based synthesis method that explicitly models the scene geometry and the projected active pattern. Furthermore, to balance the varying contributions of different supervisions during training, we design a hybrid supervision regularization strategy that dynamically adjusts the loss weights to achieve stable optimization.

## 2. Installation

### 2.1 Install PyTorch (CUDA 11.6)

```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

### 2.2 Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Training

### 3.1 Data List Format

Before using any data list in this project, make sure the `.txt` files follow the format expected by the dataloader.

- Each line in a `.txt` file describes **one sample**.
- All paths in a line are separated by **spaces**.

The number of fields depends on the data domain:

| Domain | Fields | Line format |
|--------|----------|-------------|
| `sim`  | 4        | `<left_ir> <right_ir> <depth> <disp_or_default>` |
| `real` | 7        | `<left_ir> <right_ir> <left_pattern> <right_pattern> <depth> <disp> <AO>` |

### 3.2 Configuration

Before training, edit `configs.py` to set up:

- the backbone configuration;
- the data list paths.

### 3.3 Entry Scripts

| Script   | Purpose |
|----------|---------|
| `train.py` | Train a model and save logs / checkpoints |
| `test.py`  | Evaluate a trained model and save metrics / visualizations |

### 3.4 Common Arguments

Both scripts share the following arguments:

| Argument | Choices | Description |
|----------|---------|-------------|
| `--backbone` | `stereonet`, `raft`, `psmnet` | Network backbone |
| `--datatype` | `baseline`, `ours` | Training method |
| `--gpu_id`   | any available GPU id | GPU to use |

### 3.5 Usage

```bash
python train.py \
  --backbone <BACKBONE> \
  --datatype <DATATYPE> \
  --gpu_id <GPU_ID>
```

Example:

```bash
python train.py \
  --backbone stereonet \
  --datatype baseline \
  --gpu_id 0
```

### 3.6 Outputs

After training, two folders are created in the **current working directory**:

| Folder          | Contents |
|-----------------|----------|
| `runs/`         | Training logs |
| `checkpoints/`  | Model weights |

## 4. Testing

### 4.1 Data List Format

Regardless of the data domain, each line contains **3 fields**:

| Fields | Line format |
|----------|-------------|
| 3        | `<left_ir> <right_ir> <depth>` |

### 4.2 Usage

```bash
python test.py \
  --backbone <BACKBONE> \
  --datatype <DATATYPE> \
  --gpu_id <GPU_ID> \
  --weight <PATH_TO_CHECKPOINT> \
  --test_list <PATH_TO_TEST_LIST> \
  [--save_vis]
```

### 4.3 Arguments

`--backbone`, `--datatype` and `--gpu_id` are shared with training (see [3.4 Common Arguments](#34-common-arguments)). Additional arguments:

| Argument       | Required | Description |
|----------------|----------|-------------|
| `--weight`     | Yes      | Path to the model checkpoint (`.pth`) |
| `--test_list`  | Yes      | Path to the test list `.txt` file |
| `--save_vis`   | No       | If set, save per-sample visualizations |

### 4.4 Outputs

After testing, a folder is created in the **current working directory**:

| Folder           | Contents |
|------------------|----------|
| `test_results/`  | Evaluation metrics |

If `--save_vis` is enabled, per-sample visualizations are also saved.

### 4.5 Examples

Evaluate with StereoNet on the baseline dataset:

```bash
python test.py \
  --backbone stereonet \
  --datatype baseline \
  --gpu_id <GPU_ID> \
  --weight <PATH_TO_CHECKPOINT> \
  --test_list <PATH_TO_TEST_LIST> \
  --save_vis
```

Evaluate with RAFT on the `ours` dataset without visualizations:

```bash
python test.py \
  --backbone raft \
  --datatype ours \
  --gpu_id <GPU_ID> \
  --weight <PATH_TO_CHECKPOINT> \
  --test_list <PATH_TO_TEST_LIST>
```
