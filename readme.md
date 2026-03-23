# Synergising Parameter-Efficient State Space Models with Hierarchical Attention: A Unified Framework for Multi-Source Remote Sensing Image Fusion

[![DOI](https://zenodo.org/badge/DOI/10.5281/10.5281/zenodo.19181528.svg)](https://doi.org/10.5281/zenodo.19181528)

This repository contains the official PyTorch implementation of the paper:

**“Synergising Parameter-Efficient State Space Models with Hierarchical Attention: A Unified Framework for Multi-Source Remote Sensing Image Fusion”**  
Authors: Xiandai Cui, Li Zhang *  
*The Visual Computer, 2026 (submitted)*

> **Note:** This code is directly related to the manuscript submitted to *The Visual Computer* . If you find this work useful for your research, please cite our paper (BibTeX entry provided at the end of this README).

---

## 📌 Abstract

The fusion of multi-source remote sensing imagery is critical for advancing land cover classification and environmental monitoring. While Vision Transformers excel at capturing global context, their quadratic computational complexity hinders processing of high-resolution data. Conversely, the emerging Mamba architecture offers linear efficiency, but its adaptation for multi-directional image scanning often necessitates multiple independent State Space Model (SSM) modules, increasing parameters and computation, and existing multi-scale fusion methods lack hierarchical context modeling. To address these limitations, we propose the Pre-Fusion Multi-Directional Mamba (PFMDM) framework, built on two core innovations. First, the Pre-Fusion Multi-Directional Block (PFMDB) extracts multi-directional features via parallel convolutional branches, fuses them through a gating mechanism, and processes them with a single shared SSM block, significantly reducing parameters and computation while preserving omnidirectional context. Second, the Multi-Scale Contextualized Attention (MSCA) module implements a two-stage paradigm: local multi-scale dilated convolution extraction followed by global self-attention integration, achieving progressive hierarchical context modeling from local details to global semantics. Here we show that PFMDM achieves state-of-the-art performance across three multi-source benchmarks (Muufl, University of Houston, Augsburg) with overall accuracies of 96.31%, 99.86%, and 97.80%, respectively, and also generalises robustly to unimodal hyperspectral (Indian Pines) and PolSAR (MPOLSAR) data. This work presents a unified, parameter-efficient, and context-aware fusion paradigm, offering a robust and generalisable solution for advanced multi-source remote sensing analysis.

---

## 🚀 Quick Start

### 1. Environment Setup

We recommend using Python 3.11 and Conda for environment management.

```bash
conda create -n pfdm python=3.11
conda activate pfdm
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install mamba-ssm==2.3.0  # Note: mamba-ssm may require CUDA >= 11.8
pip install -r requirements.txt 
```

**Requirements** :

```
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
tqdm
einops
timm
```

### 2. Data Preparation

The code supports the following datasets:
- **Muufl** (hyperspectral + LiDAR)
- **University of Houston** (hyperspectral + LiDAR)
- **Augsburg** (hyperspectral + SAR)
- **Indian Pines** (hyperspectral, unimodal)
- **MPOLSAR** (PolSAR, unimodal)

Please download the datasets from their official sources and organize them under the `./data/` directory as follows:

```
./data/
  ├── muufl/
  ├── houston/
  ├── augsburg/
  ├── indian_pines/
  └── mpolsar/
```

Each dataset folder should contain the original `.mat` files. 

### 3. Training

To train PFMDM on the Muufl dataset (default):

```bash
python main.py --learning_rate 1e-3 --train_bs 32
```

---

## 📁 Repository Structure

```
PFMDM/
├── Datasets/              # Dataset files
│   └── MUUFL/             # MUUFL dataset
│       ├── HSI.mat        # Hyperspectral image data of MUUFL dataset
│       ├── LiDAR.mat      # LiDAR-derived elevation data of MUUFL dataset
│       └── muufl_gt.mat   # Ground truth labels of MUUFL dataset
├── main.py                # Entry point for training and evaluation
├── model.py               # Main PFMDM architecture integrating PFMDB and MSCA
├── msca.py                # Multi-Scale Contextualized Attention (MSCA) module
├── qumamba.py             # Pre-Fusion Multi-Directional Block (PFMDB)
├── pscan.py               # Parallel scanning functions lib
├── load_data.py           # Dataset loading, preprocessing, and augmentation
├── utils.py               # Evaluation metrics (OA, AA, Kappa) and visualization utilities
├── requirements.txt       # Python dependencies
├── README.md              # Project overview and usage instructions
└── LICENSE                # MIT License
```

---

## 🔬 Key Algorithm Description

- **PFMDB (Pre-Fusion Multi-Directional Block)**:  
  Implements parallel convolutional branches to capture directional features, fuses them via a learnable gating mechanism, and passes the fused representation through a single shared SSM block (Mamba). This reduces parameters compared to using independent SSMs per direction.

- **MSCA (Multi-Scale Contextualized Attention)**:  
  A two-stage module: (1) local multi-scale features are extracted using dilated convolutions with rates; (2) global self-attention integrates these multi-scale features. This enables hierarchical context modeling from fine-grained details to global semantics.

Detailed architecture diagrams and ablation studies can be found in the paper (Section 3 and Section 4.3).

---

## 📝 Citation

If you use this code or find our work helpful, please cite:

```bibtex
@unpublished{cui2026pfmdm,
  title         = {Synergising Parameter-Efficient State Space Models with Hierarchical Attention: A Unified Framework for Multi-Source Remote Sensing Image Fusion},
  author        = {Xiandai Cui, Liping Huang, and Li Zhang},
  note          = {Submitted to The Visual Computer, under review},
  year          = {2026}
}
```

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the corresponding author.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
