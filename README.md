# Medical XPCI Anisotropic Super-Resolution / Interpolation (医疗XPCI各向异性超分辨率/插值)

## 项目简介 (Project Overview)

This project explores the application of deep learning for **anisotropic super-resolution** in the context of **UCL's advanced X-ray phase-contrast computed tomography (XPCI) technique**. XPCI offers exceptional soft tissue contrast but traditionally requires time-consuming, multi-directional sample scanning due to inherent data gaps. Our goal is to replace this mechanical, bi-directional scanning with a more efficient single-direction acquisition, leveraging deep learning to achieve a fully isotropic, high-resolution volume.

The core of this work involves a **U-Net model with a Swin Transformer encoder**, trained on the **Human Organ Atlas (LADAF-202027)** dataset. We simulate the sparse, anisotropic sampling by downsampling high-resolution 2D image patches by a factor of eight in one dimension, then linearly interpolating them back to the original size to serve as low-resolution inputs for the model to refine.

The trained model achieved a **peak signal-to-noise ratio (PSNR) of 24.07 dB** and a **structural similarity index (SSIM) of 0.63** on the Phase-contrast CT test set, significantly outperforming standard linear interpolation (PSNR 22.52 dB, SSIM 0.44) and baseline SA-INR interpolation model(PSNR 23.96 dB, SSIM 0.54). These results demonstrate the high effectiveness and feasibility of a deep learning-based approach for interpolating missing slices in XPCI data, paving the way for dramatically reduced acquisition times while preserving high-fidelity structural detail.

---

本项目旨在探索深度学习在**UCL先进X射线相位对比计算断层扫描 (XPCI) 技术**中实现**各向异性超分辨率**的应用。XPCI技术能为软组织提供卓越的对比度，但由于其设计固有的数据间隙，传统上需要耗时的多方向样本扫描。我们的目标是，用更高效的单方向采集取代这种机械的双方向扫描，并利用深度学习插值方法来获得完全各向同性的高分辨率体积。

该项目的核心是使用一个**带有Swin Transformer编码器的U-Net模型**，并在**人体器官图谱 (LADAF-202027)** 数据集上进行训练。我们通过模拟稀疏、各向异性的采样过程来创建训练数据：将高分辨率的2D图像块在某一维度上进行八倍下采样，然后通过线性插值将其恢复到原始尺寸，作为模型细化的低分辨率输入。

训练后的模型在带有大量高频细节的相位衬度CT图像测试集上取得了**24.07 dB的峰值信噪比 (PSNR)** 和**0.68的结构相似性指数 (SSIM)**，显著优于标准线性插值（PSNR 22.52 dB，SSIM 0.44）和SA-INR插值模型(PSNR 23.96 dB, SSIM 0.54)。这些结果证实，基于深度学习的方法是插值XPCI数据中缺失切片的高效可行方案，为大幅缩短采集时间同时生成高保真结构细节提供了有前景的途径。

## 核心模型架构：Swin-Unet 设计思路 (Swin-Unet Architecture Design Philosophy)

### Why Swin-Unet for this task? (为什么选择Swin-Unet？)

Traditional U-Net architectures, while effective, often rely on convolutional layers that have a limited receptive field. For complex tasks like medical image interpolation, especially with anisotropic data where long-range dependencies are crucial for accurately reconstructing intricate anatomical structures, a global understanding of the image context is beneficial. Swin-Unet addresses this by integrating the strengths of **Swin Transformer** into the U-Net framework.

传统的U-Net架构虽然有效，但通常依赖于感受野有限的卷积层。对于像医学图像插值这样复杂的任务，特别是在各向异性数据中，长距离依赖对于准确重建复杂的解剖结构至关重要，此时对图像上下文的全局理解会非常有益。Swin-Unet通过将**Swin Transformer**的优势整合到U-Net框架中来解决这个问题。

### Swin-Unet Architecture Breakdown (Swin-Unet架构拆解)

Our Swin-Unet model leverages a powerful **Swin Transformer** as its encoder, combined with a convolutional U-Net-like decoder.

我们的Swin-Unet模型利用强大的**Swin Transformer**作为其编码器，并结合了一个卷积U-Net风格的解码器。

#### 1. Swin Transformer Encoder (Swin Transformer编码器)

* **Foundation**: The encoder is built upon the **Swin Transformer** architecture (e.g., `swin_base_patch4_window7_224` from `timm`). Swin Transformer introduces **Shifted Windows** for self-attention computation, which allows for global interaction while maintaining computational efficiency by restricting self-attention to local windows.
    * **Hierarchical Feature Representation**: Unlike standard Vision Transformers, Swin Transformer builds a hierarchical feature map, similar to CNNs. This means it can extract features at multiple scales (e.g., 4x, 8x, 16x, 32x downsampling), making it suitable for U-Net's multi-scale design.
    * **Pre-trained Power**: Using a pre-trained Swin Transformer (e.g., on ImageNet) provides a robust initialization, allowing the model to leverage vast prior knowledge of general image features, which is then fine-tuned for medical image specifics.
    * **Medical Image Context**: The ability of Swin Transformer to capture both local detail within windows and global context through shifted windows is particularly advantageous for medical images, where both fine textures (e.g., cell structures, vessel walls) and broader anatomical relationships are critical.

* **基础**: 编码器建立在**Swin Transformer**架构之上（例如，`timm`库中的`swin_base_patch4_window7_224`）。Swin Transformer引入了**移位窗口（Shifted Windows）**进行自注意力计算，通过将自注意力限制在局部窗口内，实现了全局交互，同时保持了计算效率。
    * **分层特征表示**: 与标准Vision Transformer不同，Swin Transformer构建了类似于CNN的分层特征图。这意味着它可以在多个尺度（例如，4倍、8倍、16倍、32倍下采样）提取特征，使其非常适合U-Net的多尺度设计。
    * **预训练能力**: 使用预训练的Swin Transformer（例如，在ImageNet上）提供了强大的初始化，使模型能够利用大量的通用图像特征先验知识，然后针对医学图像的特定性进行微调。
    * **医学图像上下文**: Swin Transformer在窗口内捕获局部细节和通过移位窗口捕获全局上下文的能力，对于医学图像尤其有利，因为精细纹理（如细胞结构、血管壁）和更广泛的解剖关系都至关重要。

#### 2. U-Net-like Decoder (U-Net风格的解码器)

* **Upsampling Path**: The decoder reverses the encoding process, gradually upsampling the feature maps back to the original input resolution (or desired output resolution). We primarily use **PixelShuffle (sub-pixel convolution)** for upsampling. PixelShuffle is a highly effective and learnable upsampling method that can generate smoother outputs with fewer checkerboard artifacts compared to transposed convolutions, making it ideal for high-fidelity image reconstruction.
* **Skip Connections**: Crucial for U-Net type architectures, skip connections directly link feature maps from the encoder to the corresponding resolution levels in the decoder. This allows the decoder to recover fine-grained spatial information lost during the downsampling process, enabling the reconstruction of sharp edges and detailed textures.
* **Convolutional Blocks**: The upsampled and concatenated feature maps are processed by standard convolutional blocks (e.g., `DoubleConv` with `Conv2d`, `BatchNorm2d`, `ReLU`) to refine the features at each resolution level.

* **上采样路径**: 解码器反转编码过程，逐步将特征图上采样回原始输入分辨率（或所需的输出分辨率）。我们主要使用**PixelShuffle（子像素卷积）**进行上采样。与转置卷积相比，PixelShuffle是一种高效且可学习的上采样方法，可以生成更平滑、棋盘格伪影更少的输出，非常适合高保真图像重建。
* **跳跃连接**: 对于U-Net型架构至关重要，跳跃连接直接将编码器中的特征图连接到解码器中对应的分辨率层。这使得解码器能够恢复在下采样过程中丢失的精细空间信息，从而重建清晰的边缘和详细的纹理。
* **卷积块**: 经过上采样和拼接的特征图由标准卷积块（例如，包含`Conv2d`、`BatchNorm2d`、`ReLU`的`DoubleConv`）处理，以在每个分辨率级别上细化特征。

#### 3. Loss Function (损失函数)

To achieve high-fidelity reconstruction, our model is trained using a composite loss function:

* **Mean Squared Error (MSE) / L2 Loss**: Measures the pixel-wise difference between the predicted and ground truth images.
* **High-Frequency (Laplacian) Loss**: To explicitly encourage the model to restore sharp edges and fine details, we incorporate a Laplacian-based high-frequency loss. This penalizes differences in the high-frequency components of the images, which are critical for perceiving detail in medical scans.

为了实现高保真重建，我们的模型使用复合损失函数进行训练：

* **均方误差（MSE）/ L2损失**: 衡量预测图像和真实图像之间的像素级差异。
* **高频（拉普拉斯）损失**: 为了明确鼓励模型恢复清晰的边缘和精细细节，我们引入了基于拉普拉斯算子的高频损失。这惩罚了图像高频分量中的差异，这对于感知医学扫描中的细节至关重要。
