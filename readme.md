# PFMDM: Pre-Fusion Multi-Directional SSM for Multi-source Remote Sensing Image Fusion

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19181528.svg)](https://doi.org/10.5281/zenodo.19181528)

This repository contains the official PyTorch implementation of the paper:

**“PFMDM: Pre-Fusion Multi-Directional SSM for Multi-source Remote Sensing Image Fusion”**  
Authors: Xiandai Cui, Liping Huang, Li Zhang *  
*The Visual Computer, 2026 (submitted)*

> **Note:** This code is directly related to the manuscript submitted to *The Visual Computer* . If you find this work useful for your research, please cite our paper (BibTeX entry provided at the end of this README).

---

## 📌 Abstract

Fusing remote sensing images from multiple sources significantly improves the accuracy and effectiveness of land cover classification and environmental monitoring. Current mainstream methods face two major limitations. First, Vision Transformers (ViTs) can capture global contextual information, but their computational complexity grows quadratically with input size. This makes them inefficient for processing high-resolution remote sensing images. Second, while the emerging Mamba model offers linear complexity, its multi-directional scanning in the image domain usually requires multiple independent State Space Model (SSM) modules. This leads to a significant increase in both parameters and computational cost. Moreover, existing multi-scale fusion methods lack a hierarchical contextual modeling mechanism that progressively integrates information from local to global levels. To address these issues, this paper proposes the Pre-Fusion Multi-Directional Mamba (PFMDM) framework. Its core innovations are: (1) the design of the Pre-Fusion Multi-Directional Block (PFMDB), which first extracts multi-directional features through parallel convolutional branches, then fuses them via a gating mechanism before feeding them into a single shared Mamba block, significantly reducing parameter and computational complexity while preserving the integrity of multi-directional information; and (2) the introduction of the Multi-Scale Contextualized Attention (MSCA) module, which implements a two-stage paradigm of "local multi-scale dilated convolution extraction" followed by "global self-attention integration," achieving progressive hierarchical contextual modeling from local details to global semantics. The proposed approach was tested on Muufl, the University of Houston, and the Augsburg datasets, yielding overall accuracy rates of 96.31%, 99.86%, and 97.80%, respectively. Additional experiments were performed on unimodal datasets, including Indian Pines (hyperspectral) and MPOLSAR (PolSAR). The results indicate that PFMDM offers a unified framework for multi-source remote sensing image fusion that is parameter-efficient, sensitive to contextual information, and robust across modalities.

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

Please download the datasets from their official sources and organize them under the `./Datasets/` directory as follows:

```
./Datasets/
  ├── MUUFL/
  ├── houston/
  ├── Augsburg/
  ├── Indian_pines/
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
  title         = {PFMDM: Pre-Fusion Multi-Directional SSM for Multi-source Remote Sensing Image Fusion},
  author        = {Cui, Xiandai and Huang, Liping and Zhang, Li},
  note          = {Submitted to Signal, Image and Video Processing, under review},
  year          = {2026}
}
```

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the corresponding author.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
