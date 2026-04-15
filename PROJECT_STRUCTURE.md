# CAME-Net 项目结构

## 项目概述
CAME-Net (Clifford Attention Multimodal Equivariant Network) 是一个基于几何代数 (PGA) 的点云分类网络，具有几何等变性和多模态融合能力。

## 主要文件结构

### 核心模块 (主路径)

#### `pga_algebra.py` - 几何代数基础
- **职责**: 实现 Projective Geometric Algebra (PGA) 的核心运算
- **关键功能**:
  - 多向量 (Multivector) 类定义
  - 几何乘积 (geometric product)
  - 电机 (motor) 生成和应用
  - 刚性变换 (旋转和平移)
- **导出**: `GRADE_INDICES` (统一的等级索引)

#### `gln.py` - 等级归一化
- **职责**: 提供统一的等级归一化实现
- **关键功能**:
  - `GradewiseLayerNorm` - 基于 RMS 的等级归一化
  - 支持可学习的尺度和偏置
  - 保持几何结构的稳定性
- **设计原则**: 这是唯一的归一化实现源，其他模块都从这里导入

#### `mpe.py` - 多模态投影嵌入
- **职责**: 将不同模态数据转换为 PGA 多向量表示
- **关键功能**:
  - `PointCloudMPE` - 点云嵌入
  - `ImageMPE` - 图像嵌入
  - `TextMPE` - 文本嵌入
  - `MultimodalMPE` - 多模态融合
- **输出保证**: 稳定输出 (B, N, 16) 的多向量

#### `gca.py` - 几何克利福德注意力
- **职责**: 实现几何感知的注意力机制
- **关键功能**:
  - `GeometricCliffordAttention` - 基于标量部分的注意力
  - 等级感知的注意力权重
  - 电机值注意力 (可选)
- **设计特点**: 使用统一的 `GradewiseLayerNorm`

#### `came_net.py` - 主网络架构
- **职责**: 完整的 CAME-Net 网络实现
- **关键功能**:
  - `CAMELayer` - 网络层实现
  - `CAMENet` - 主网络类
  - 前向传播和嵌入提取
  - 潜在多向量输出接口
- **设计原则**: 统一的归一化导入和标准的残差连接顺序

#### `equiv_loss.py` - 等变损失
- **职责**: 实现几何等变性损失函数
- **关键功能**:
  - `equivariance_loss_efficient` - 高效的等变损失
  - 基于 Φ(M·X) vs M·Φ(X) 的损失计算
  - 在点级潜在多向量上计算损失
- **设计特点**: 使用统一的 `GRADE_INDICES`

### 辅助模块 (非主路径)

#### `data_utils.py` - 数据工具
- **职责**: 数据加载和预处理
- **功能**: 随机点云数据集、数据增强、批处理函数

#### `train.py` - 训练脚本
- **职责**: 模型训练和验证
- **功能**: 训练循环、检查点保存、模型评估

#### `test_came_net.py` - 测试脚本
- **职责**: 验证所有核心功能
- **覆盖范围**: 多向量运算、MPE 模块、GCA 层、前向传播、反向传播、等变损失

## 模块依赖关系

```
主路径依赖:
came_net.py → gca.py, mpe.py, gln.py, pga_algebra.py
gca.py → gln.py, pga_algebra.py  
mpe.py → pga_algebra.py
gln.py → pga_algebra.py

统一导出:
pga_algebra.py → GRADE_INDICES
gln.py → GradewiseLayerNorm
```

## 设计原则

1. **统一性**: 所有模块使用统一的 `GRADE_INDICES` 和 `GradewiseLayerNorm`
2. **稳定性**: 保证输出形状的稳定性，特别是 (B, N, 16)
3. **几何保持**: 所有操作都设计为保持几何结构
4. **模块化**: 清晰的职责分离和模块化设计

## 测试覆盖

测试脚本验证以下核心功能:
- 多向量基本运算
- 各 MPE 模块的输出形状
- GCA 层的有限值输出
- 网络前向传播
- 一步反向传播
- 等变损失计算
- 等级索引完整性
- 潜在多向量提取

## 使用示例

```python
from came_net import CAMENet
from equiv_loss import equivariance_loss_efficient

# 创建模型
model = CAMENet(num_classes=40, num_layers=4, num_heads=8)

# 前向传播
point_coords = torch.randn(2, 1024, 3)
logits = model(point_coords)

# 提取嵌入
embedding = model.get_point_cloud_embedding(point_coords)
latent_mv = model.get_latent_multivector(point_coords)

# 计算等变损失
loss = equivariance_loss_efficient(model, point_coords, None)
```

## 注意事项

- 所有模块都设计为与 PyTorch 兼容
- 几何运算使用 PGA(3,0,1) 规范
- 多向量表示为 16 维张量
- 网络设计为几何等变的，对刚性变换具有鲁棒性