## Synergising Parameter-Efficient State Space Models with Hierarchical Attention: A Unified Framework for Multi-Source Remote Sensing Image Fusion

## Abstract

The fusion of multi-source remote sensing imagery is critical for advancing land cover classification and environmental monitoring. While Vision Transformers excel at capturing global context, their quadratic computational complexity hinders processing of high-resolution data. Conversely, the emerging Mamba architecture offers linear efficiency, but its adaptation for multi-directional image scanning often necessitates multiple independent State Space Model (SSM) modules, increasing parameters and computation, and existing multi-scale fusion methods lack hierarchical context modeling. To address these limitations, we propose the Pre-Fusion Multi-Directional Mamba (PFMDM) framework, built on two core innovations. First, the Pre-Fusion Multi-Directional Block (PFMDB) extracts multi-directional features via parallel convolutional branches, fuses them through a gating mechanism, and processes them with a single shared SSM block, significantly reducing parameters and computation while preserving omnidirectional context. Second, the Multi-Scale Contextualized Attention (MSCA) module implements a two-stage paradigm: local multi-scale dilated convolution extraction followed by global self-attention integration, achieving progressive hierarchical context modeling from local details to global semantics. Here we show that PFMDM achieves state-of-the-art performance across three multi-source benchmarks (Muufl, University of Houston, Augsburg) with overall accuracies of 96.31%, 99.86%, and 97.80%, respectively, and also generalises robustly to unimodal hyperspectral (Indian Pines) and PolSAR (MPOLSAR) data. This work presents a unified, parameter-efficient, and context-aware fusion paradigm, offering a robust and generalisable solution for advanced multi-source remote sensing analysis. 

## Requirements:

- Python 3.11
- PyTorch >= 2.10.0
- mamba-ssm>=2.3.0

## Usage:

python main.py

