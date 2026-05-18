# Physics-Informed Wavelet–Fourier Model for Multiscale Nonlinear PDEs

This repository contains the official implementation of the paper:  
**"Physics-informed wavelet–Fourier model for multiscale nonlinear partial differential equations"**

The proposed **PIWF** model is a physics-informed unsupervised learning framework that accurately solves multiscale nonlinear PDEs with discontinuities and complex structures. It integrates a **wavelet layer** for multiscale local feature extraction, a **Fourier-branch** for global frequency representation, a **residual MLP** for nonlinear mapping, and a **channel attention** mechanism for adaptive feature fusion.

## 📌 Highlights

- 🌊 **Wavelet layer** – captures sharp gradients and discontinuities via multilevel wavelet series approximation  
- 🌍 **Fourier branch** – maintains global spectral consistency  
- 🧠 **Residual MLP** – enhances expressivity for complex nonlinear mappings  
- 🔗 **Channel attention** – adaptively balances local and global features  
- 📉 **Physics-informed training** – no labeled data required, loss function encodes PDE residuals  
- 🔬 Validated on **Burgers’ equation**, **Shallow water equations**, and **2D cylinder wake flow**  
- 📊 Compared with **PINN**, **PIKAN**, and **FD** (finite difference), including ablation studies


Each model folder contains the complete implementation for that method, including training scripts, configuration files, and result visualization.

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.10+
- Other dependencies: `numpy`, `matplotlib`, `scipy`, `pyyaml`, `tqdm`

### Run an example

To train/test the PIWF model on the 2D cylinder wake flow problem, execute:
cd PIWF/cylinder
python cylinder_PIWF.py
📊 Reproducing Paper Results
﻿
All experimental results reported in the paper can be reproduced by running the provided scripts with default hyperparameters
