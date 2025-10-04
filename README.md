# XPCI-DL-Interpolation: Deep Learning for Anisotropic X-ray Phase-Contrast CT Data Interpolation

![Project Banner](https://img.shields.io/badge/Status-Ongoing-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Python Version](https://img.shields.io/badge/Python-3.8+-blueviolet)
![PyTorch Version](https://img.shields.io/badge/PyTorch-1.8+-orange)

## 目录 (Table of Contents)
1.  [项目简介 (Project Overview)](#1-项目简介-project-overview)
2.  [学术背景 (Academic Background)](#2-学术背景-academic-background)
3.  [方法论 (Methodology)](#3-方法论-methodology)
    * [数据模拟与处理 (Data Simulation & Preprocessing)](#数据模拟与处理-data-simulation--preprocessing)
    * [Swin-Unet 模型架构 (Swin-Unet Model Architecture)](#swin-unet-模型架构-swin-unet-model-architecture)
    * [训练策略 (Training Strategy)](#训练策略-training-strategy)
4.  [Swin-Unet 架构设计思路与优化方向 (Swin-Unet Architecture Design & Optimization)](#4-swin-unet-架构设计思路与优化方向-swin-unet-architecture-design--optimization)
5.  [结果 (Results)](#5-结果-results)
6.  [未来工作 (Future Work)](#6-未来工作-future-work)
7.  [安装与使用 (Installation & Usage)](#7-安装与使用-installation--usage)
8.  [引用 (Citation)](#8-引用-citation)
9.  [贡献 (Contributing)](#9-贡献-contributing)
10. [许可证 (License)](#10-许可证-license)

---

## 1. 项目简介 (Project Overview)

本项目旨在通过深度学习方法，解决 UCL 先进的 X 射线相衬计算机断层扫描 (XPCI) 技术中固有的数据稀疏问题。通过开发并训练一个基于 Swin Transformer 的 U-Net 模型，我们探索了用计算插值替代耗时的机械双向扫描的可行性，以期实现快速、各向同性的高分辨率体数据重建。

This project aims to address the inherent data sparsity in UCL's advanced X-ray Phase-Contrast Computed Tomography (XPCI) technique using deep learning. By developing and training a U-Net model with a Swin Transformer encoder, we investigate the feasibility of replacing time-consuming mechanical bi-directional scanning with computational interpolation, aiming for rapid, isotropic, high-resolution volume reconstruction.

---

## 2. 学术背景 (Academic Background)

UCL 的先进 X 射线相衬计算机断层扫描 (XPCI) 技术为软组织提供了卓越的对比度，但其设计固有的数据缺失需要耗时且多方向的样本扫描来填补数据空白。[14] 这些数据空白导致采集过程耗时，特别是在需要高分辨率各向同性体积时。本项目研究用更高效的**单向采集**取代机械式双向扫描，并通过**深度学习插值方法**实现完全各向同性的高分辨率体数据。

UCL’s advanced X-ray Phase-Contrast Computed Tomography (XPCI) technique provides exceptional contrast for soft tissues but requires time-consuming, multi-directional sample scanning to fill data gaps inherent to its design.[14] These data gaps lead to prolonged acquisition times, especially when a high-resolution isotropic volume is desired. This project investigates replacing the mechanical, bi-directional scanning with a more efficient **single-direction acquisition**, using a **deep learning interpolation method** to achieve a fully isotropic, high-resolution volume.

---

## 3. 方法论 (Methodology)

### 数据模拟与处理 (Data Simulation & Preprocessing)

* **数据集**: 人体器官图谱 (Human Organ Atlas, LADAF-202027) 数据集。
* **稀疏采样模拟**: 为了模拟 XPCI 数据的稀疏、各向异性采样过程，我们对高分辨率 2D 图像块进行了处理：
    * 将 256x256 的高分辨率图像沿一个维度（模拟 Z 轴）下采样了八倍，得到 256x32 的低分辨率图像。
    * 随后，通过线性插值（双线性插值）将 256x32 的图像恢复到 256x256，作为模型的低分辨率输入 (`LR_interpolated`)，以供模型进一步细化。
    * 模型的真实高分辨率输出 (`HR_ground_truth`) 则是原始的 256x256 图像。
    * **输入**: `[B, 1, 256, 256]` (线性插值后的低分辨率图像)
    * **目标**: `[B, 1, 256, 256]` (原始高分辨率图像)

### Swin-Unet 模型架构 (Swin-Unet Model Architecture)

我们采用了基于 **Swin Transformer** 编码器的 **U-Net** 模型。

* **编码器 (Encoder)**: 使用预训练的 Swin Transformer 作为骨干。
    * 初始版本使用 `swin_tiny_patch4_window7_224`。
    * 后续优化尝试升级到 `swin_base_patch4_window7_224` 以增强特征提取能力。
    * Swin Transformer 能够有效地捕获图像中的局部和全局上下文信息，特别适合处理医疗影像中复杂的解剖结构。
    * 由于 Swin Transformer 预训练通常需要 3 通道输入，对于单通道灰度图，会在输入模型前将其复制为 3 通道。
* **解码器 (Decoder)**: 沿用 U-Net 的经典结构，采用**跳跃连接 (Skip Connections)** 将编码器不同阶段的特征与解码器进行融合，以保留空间细节。
    * 上采样模块使用 **PixelShuffle**（子像素卷积），能够学习更平滑、更少伪影的上采样。
    * 解码器层由 `DoubleConv` 模块组成，包含 `Conv2d -> BatchNorm2d -> ReLU` 序列。
* **输出**: 最终通过一个 1x1 卷积层输出单通道的插值高分辨率图像。

### 训练策略 (Training Strategy)

* **损失函数**: 结合了 **均方误差 (MSELoss)** 和 **高频损失 (High-Frequency Loss)**。高频损失通过拉普拉斯核计算，旨在确保模型能更好地恢复图像的细节和边缘，对于医疗影像至关重要。
    * $L_{total} = MSE Loss(Output, Target) + \lambda_{hf} \times High\_Frequency\_Loss(Output, Target)$
    * 其中 $\lambda_{hf}$ 是高频损失的权重，本项目设置为 0.5。
* **优化器**: Adam 优化器。
* **学习率**: 初始学习率为 1e-4。
* **数据增强**: 包含随机旋转（90度）以增加训练样本的多样性。
* **批次大小 (Batch Size)**: 8。
* **训练轮次 (Epochs)**: 15。

---

## 4. Swin-Unet 架构设计思路与优化方向 (Swin-Unet Architecture Design & Optimization)

Swin-Unet 的设计巧妙地结合了 Transformer 的全局感受野优势和 U-Net 在图像到图像转换任务中的强大空间保留能力。

**设计思路 (Design Philosophy):**

* **Transformer 的引入**: 传统的 CNN 在处理长距离依赖（全局上下文）方面存在局限性。Swin Transformer 通过**移动窗口机制 (Shifted Window Mechanism)** 和**分层设计 (Hierarchical Design)**，在保持计算效率的同时，有效捕获不同尺度的全局信息。这对于理解医疗影像中复杂的解剖结构（如器官边界、病灶区域）及其与周围组织的关联至关重要。
* **U-Net 的保留**: U-Net 架构以其编码器-解码器结构和**跳跃连接**而闻名，能够有效地进行多尺度特征融合，将编码器提取的深层语义信息与浅层高分辨率空间信息结合起来。这对于生成高精度的细节丰富的插值图像是不可或缺的。
* **PixelShuffle 的上采样**: 使用 **PixelShuffle** 代替传统的反卷积（Transposed Convolution）或双线性插值，使上采样过程可学习，能够产生更平滑、更少棋盘格伪影的输出，对于图像质量至关重要。

**优化方向 (Optimization Directions):**

1.  **Swin Transformer 骨干升级**:
    * **当前**: `swin_tiny` -> `swin_base`。
    * **未来**: 进一步升级到 `swin_large` 或 `swin_giant` 版本，以利用更强大的预训练特征提取能力。但这会显著增加模型参数和计算资源需求。
2.  **定制化上采样模块**:
    * 当前的上采样是针对 256x256 -> 256x256 图像到图像的任务。
    * **优化**: 重新设计网络，使其直接接收原始的 **1x256x32** 低分辨率图像。网络内部应包含专门的**各向异性 PixelShuffle 模块**（例如 `PixelShuffle(scale_factor=(8, 1))` 或多个 `PixelShuffle(scale_factor=(1, 2))` 组合）来在高度维度进行 8 倍上采样。这样能更好地利用 Swin Transformer 在低分辨率特征图上提取的特征，避免线性插值引入的伪影。
3.  **损失函数优化**:
    * **感知损失 (Perceptual Loss)**: 结合基于预训练 VGG 等网络的感知损失，能更好地衡量生成图像在特征空间上的相似度，从而生成视觉上更令人满意的细节。
    * **对抗性损失 (Adversarial Loss)**: 引入 GAN 的判别器，通过对抗性训练迫使生成器生成更逼真、更接近真实高分辨率图像的纹理。这通常能显著提高 SSIM，但训练会更复杂且可能引入“幻觉”。
4.  **数据增强策略**:
    * 在稀疏采样模拟阶段，考虑引入更复杂的模拟方式，例如模拟真实的 XPCI 噪声和伪影。
    * 除了旋转，可以加入随机缩放、弹性形变、随机亮度/对比度调整等，以提高模型的鲁棒性和泛化能力。
5.  **模型轻量化**:
    * 如果计算资源受限，可以探索 Swin Transformer 的轻量级变体（如 Deformable Swin Transformer）或使用更高效的 CNN 模块（如 GhostNet, MobileNet 的思想）与 Swin Transformer 混合，以达到性能与效率的平衡。
6.  **多阶段/级联方法**:
    * 可以考虑训练一个**两阶段模型**：第一阶段生成一个初步的插值图像，第二阶段接收第一阶段的输出和原始 LR 图像，进一步细化和恢复细节。
7.  **3D 建模**:
    * 如果 XPCI 数据本身是 3D 体积且需要 3D 插值，可以考虑扩展 Swin-Unet 到 **3D Swin-Unet**，直接处理 3D 体素数据，但这将对计算资源提出更高要求。

---

## 5. 结果 (Results)

经过训练，基于 Swin Transformer 编码器的 U-Net 模型表现出合理的性能：

* **峰值信噪比 (PSNR)**: 24.07 dB
* **结构相似性指数 (SSIM)**: 0.58

这些结果显著优于标准的线性插值方法：

* **线性插值 PSNR**: 22.52 dB
* **线性插值 SSIM**: 0.44

**结论**: 这些结果证实，基于深度学习的方法是插值 XPCI 数据中缺失切片的一种高效且可行的方法，为大幅缩短采集时间同时生成高保真结构细节提供了有前景的途径。

---

## 6. 未来工作 (Future Work)

* 探索更强大的 Swin Transformer 骨干（如 Swin-Large）或尝试其他 Transformer 架构。
* 引入感知损失和对抗性损失以进一步提升视觉质量。
* 优化模型架构，使其直接处理原始低分辨率输入，而非预插值输入。
* 在更大的、更多样化的医疗影像数据集上进行训练和验证。
* 将模型部署到实际的 XPCI 采集流程中进行实时验证。

---

## 7. 安装与使用 (Installation & Usage)

1.  **克隆仓库 (Clone the repository)**:
    ```bash
    git clone [https://github.com/yourusername/XPCI-DL-Interpolation.git](https://github.com/yourusername/XPCI-DL-Interpolation.git)
    cd XPCI-DL-Interpolation
    ```

2.  **创建虚拟环境并安装依赖 (Create a virtual environment and install dependencies)**:
    ```bash
    conda create -n xpcidl python=3.9
    conda activate xpcidl
    pip install -r requirements.txt
    ```
    *`requirements.txt` 示例内容:*
    ```
    torch>=1.8.0
    torchvision>=0.9.0
    timm>=0.4.12
    numpy>=1.20.0
    opencv-python>=4.5.0
    Pillow>=8.0.0
    scikit-learn>=0.24.0
    matplotlib>=3.3.0
    ```

3.  **下载数据集 (Download Dataset)**:
    * 请从 [Human Organ Atlas (LADAF-202027)](https://example.com/ladfa-dataset-link) 下载数据集，并将其放置在项目根目录下的 `data/LADAF-202027` 路径。
    * 确保您的 `patches_folder` 变量指向正确的路径。

4.  **训练模型 (Train the model)**:
    ```bash
    python train.py
    ```
    *(请将 `train.py` 替换为您的实际训练脚本名称)*

5.  **评估模型 (Evaluate the model)**:
    ```bash
    python evaluate.py --model_path path/to/your/best_model.pth
    ```
    *(请将 `evaluate.py` 替换为您的实际评估脚本名称)*

---

## 8. 引用 (Citation)

如果您在研究中使用了本项目，请考虑引用以下相关工作和本仓库：

```bibtex
@article{chen2018deep,
  title={Deep learning-based anisotropic brain MR image interpolation},
  author={Chen, Wu and Wang, Kai and Liu, Yuan and Li, Yiran and Wang, Dong and Zhang, Jianfeng and Wu, Jianhua},
  journal={Medical Image Analysis},
  volume={43},
  pages={107--119},
  year={2018},
  publisher={Elsevier}
}

@article{liu2021swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}

% 请在此处添加您项目的引用信息，例如：
% @misc{yourprojectname2023,
%   author = {Your Name/Team Name},
%   title = {XPCI-DL-Interpolation: Deep Learning for Anisotropic X-ray Phase-Contrast CT Data Interpolation},
%   year = {2023},
%   publisher = {GitHub},
%   journal = {GitHub repository},
%   howpublished = {\url{[https://github.com/yourusername/XPCI-DL-Interpolation](https://github.com/yourusername/XPCI-DL-Interpolation)}},
% }
