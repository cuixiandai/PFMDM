## PFMDM: Pre-Fusion Multi-Directional SSM for Multi-source Remote Sensing Image Fusion

## Abstract

Fusing remote sensing images from multiple sources significantly improves the accuracy and effectiveness of land cover classification and environmental monitoring. Current mainstream methods face two major limitations. First, Vision Transformers (ViTs) can capture global contextual information, but their computational complexity grows quadratically with input size. This makes them inefficient for processing high-resolution remote sensing images. Second, while the emerging Mamba model offers linear complexity, its multi-directional scanning in the image domain usually requires multiple independent State Space Model (SSM) modules. This leads to a significant increase in both parameters and computational cost. Moreover, existing multi-scale fusion methods lack a hierarchical contextual modeling mechanism that progressively integrates information from local to global levels. To address these issues, this paper proposes the Pre-Fusion Multi-Directional Mamba (PFMDM) framework. Its core innovations are: 1) the design of the Pre-Fusion Multi-Directional Block (PFMDB), which first extracts multi-directional features through parallel convolutional branches, then fuses them via a gating mechanism before feeding them into a single shared Mamba block, significantly reducing parameter and computational complexity while preserving the integrity of multi-directional information; and 2) the introduction of the Multi-Scale Contextualized Attention (MSCA) module, which implements a two-stage paradigm of "local multi-scale dilated convolution extraction" followed by "global self-attention integration," achieving progressive hierarchical contextual modeling from local details to global semantics. The proposed approach was tested on Muufl, the University of Houston, and the Augsburg datasets, yielding overall accuracy rates of 96.31%, 99.86%, and 97.80%, respectively. Additional experiments were performed on unimodal datasets, including Indian Pines (hyperspectral) and MPOLSAR (PolSAR). The results indicate that PFMDM offers a unified framework for multi-source remote sensing image fusion that is parameter-efficient, sensitive to contextual information, and robust across modalities. 

## Requirements:

- Python 3.7
- PyTorch >= 1.12.1
- mamba-ssm>=2.3.0

## Usage:

python main.py

