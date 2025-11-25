# WideTopo

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

WideTopo 是一种创新的神经网络剪枝方法，旨在通过保留训练动态和探索宽拓扑结构来提高剪枝后子网络的性能。该方法结合了神经切线核（NTK）理论和隐式目标对齐（ITA），并采用密度感知的显著性分数衰减策略和重复掩码恢复策略，以保持子网络的宽度。

## 摘要

Foresight 神经网络剪枝方法因其节省计算资源的潜力而备受关注。现有方法主要分为基于显著性分数和基于图论的方法。然而，单独依赖显著性分数可能导致深但窄的子网络，而基于图论的方法可能不适用于需要预训练参数初始化的神经网络，尤其是在迁移学习场景中。WideTopo 通过结合 NTK 理论和 ITA，捕获子网络的训练动态，并采用密度感知的显著性分数衰减策略和重复掩码恢复策略，保留更有效的节点，从而在各层中保持子网络的宽度。我们在多种模型密度率下对 CNN 和 ViT 模型进行了广泛验证，结果表明 WideTopo 在随机和预训练初始化设置下均表现出色。

## 特性

- **训练动态保留**：结合 NTK 理论和 ITA 捕获子网络的训练动态。
- **宽拓扑探索**：通过密度感知的显著性分数衰减策略和重复掩码恢复策略，保持子网络的宽度。
- **广泛验证**：在多种数据集和网络架构上验证了方法的有效性。

## 安装

1. 克隆仓库：
   ```bash
   git clone https://github.com/Memoristor/WideTopo.git
   cd WideTopo
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

### 数据准备

请将数据集下载并放置在 `datasets/` 目录下。支持的数据集包括：
- CIFAR-10/100
- ImageNet
- Pascal Context
- Tiny ImageNet

### 运行示例

1. 训练并剪枝模型：
   ```bash
   python main.py --config configs/foresight_pruning/cifar100/example_config.yaml
   ```

2. 评估模型：
   ```bash
   python training_dynamics.py --evaluate --config configs/foresight_pruning/cifar100/example_config.yaml
   ```

## 项目结构

- `configs/`：存储剪枝和训练的配置文件。
- `datasets/`：数据集加载器和相关代码。
- `engines/`：训练和剪枝引擎。
- `models/`：支持的神经网络模型。
- `pruners/`：剪枝算法实现。
- `tools/`：辅助工具函数。

## 引用

如果您在研究中使用了 WideTopo，请引用我们的论文：

```
@article{widetopo2025,
  title={WideTopo: Improving Foresight Neural Network Pruning through Training Dynamics Preservation and Wide Topologies Exploration},
  author={Memoristor},
  journal={Neural Networks},
  year={2025},
  url={https://www.sciencedirect.com/science/article/pii/S0893608025010160}
}
```

## 许可证

该项目基于 MIT 许可证开源，详情请参阅 [LICENSE](LICENSE)。

## 联系方式

如有任何问题，请通过 [GitHub Issues](https://github.com/Memoristor/WideTopo/issues) 联系我们。
