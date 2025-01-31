# PCBA表面缺陷检测系统

基于深度学习的PCB板焊接缺陷检测系统，支持多种先进的检测算法。本项目使用PCB-AoI数据集进行训练和评估，目前已完成YOLOv8和U-Net模型的基础训练和测试。

## 功能特点

- 支持多种深度学习模型（YOLOv8、U-Net等）
- 支持PCB-AoI数据集的训练和评估
- 提供完整的训练、验证和推理流程
- 可视化工具支持
- 支持CPU和GPU训练

## 项目结构

```
├── configs/           # 配置文件目录
│   ├── yolov8.yaml   # YOLOv8模型配置
│   └── unet.yaml     # U-Net模型配置
├── data/             # 数据集目录
│   ├── dataset/      # PCB-AoI数据集
│   └── pcb-aoi.md   # 数据集说明文档
├── models/           # 模型定义
│   ├── unet/        # U-Net模型实现
│   └── yolo/        # YOLOv8相关代码
├── tools/            # 工具脚本
│   ├── convert_voc_to_yolo.py  # VOC转YOLO格式工具
│   └── convert_voc_to_mask.py  # VOC转分割掩码工具
├── utils/            # 通用工具函数
│   ├── augmentation.py  # 数据增强
│   └── data_utils.py    # 数据处理
├── train.py         # 训练脚本
├── evaluate.py      # 评估脚本
├── predict.py       # YOLO预测脚本
├── predict_unet.py  # UNET预测脚本
├── requirements.txt  # 依赖包
└── README.md        # 项目说明
```

## 环境配置

1. Python环境要求：
   - Python 3.8+
   - pip3或conda包管理器

2. 安装依赖：
```bash
pip3 install -r requirements.txt
```

主要依赖包版本：
- torch >= 2.0.0
- torchvision >= 0.15.0
- ultralytics >= 8.0.0
- segmentation-models-pytorch >= 0.3.0
- albumentations >= 1.3.0
- opencv-python >= 4.7.0

## 数据准备

1. 下载PCB-AoI数据集：
   - [Kaggle链接](https://www.kaggle.com/datasets/kubeedgeianvs/pcb-aoi)
   - [Huawei OBS链接](https://kubeedge.obs.cn-north-1.myhuaweicloud.com:443/ianvs/pcb-aoi/dataset.zip)

2. 解压数据集到 `data/dataset` 目录

3. 转换数据格式：
```bash
# 转换为YOLO格式（目标检测）
python3 tools/convert_voc_to_yolo.py

# 转换为掩码格式（语义分割）
python3 tools/convert_voc_to_mask.py
```

## 使用方法

### 1. 训练模型

```bash
# 训练YOLOv8
python3 train.py --config configs/yolov8.yaml --device cpu  # 使用CPU训练
# 或
python3 train.py --config configs/yolov8.yaml --device cuda  # 使用GPU训练

# 训练U-Net
python3 train.py --config configs/unet.yaml --device cpu  # 使用CPU训练
# 或
python3 train.py --config configs/unet.yaml --device cuda  # 使用GPU训练
```

### 2. 预测

```bash
# YOLOv8预测
python3 predict.py --weights runs/train/exp/weights/best.pt --source data/dataset/test_data/images --device cpu

# U-Net预测
python3 predict_unet.py --weights runs/train/exp/weights/best.pt --source data/dataset/test_data/images --device cpu
```

## 实验结果

### 1. YOLOv8n模型（10轮训练）

- 训练设备：CPU (Apple M1 Pro)
- 训练时间：0.096小时（约5.8分钟）
- 数据集：
  - 训练集：173张图像
  - 测试集：60张图像

性能指标：
- mAP50: 0.562 (56.2%)
- mAP50-95: 0.177 (17.7%)
- Precision: 0.53 (53%)
- Recall: 0.641 (64.1%)

预测性能：
- 平均推理时间：83.3ms/图像
- 预处理时间：2.1ms/图像
- 后处理时间：0.4ms/图像

### 2. U-Net模型（3轮快速训练）

- 训练设备：CPU (Apple M1 Pro)
- 训练时间：约8-9分钟
- 数据集：
  - 训练集：173张图像
  - 测试集：60张图像
- 模型配置：
  - 编码器：ResNet34
  - 预训练权重：ImageNet
  - 输入尺寸：416x416
  - Batch Size：4

训练过程：
- 最终训练损失：0.6961
- 最终验证损失：0.6954

评估指标：
- IoU: 0.0058 (0.58%)
- 精确率: 0.0058 (0.58%)
- 召回率: 1.0000 (100%)
- F1分数: 0.0115 (1.15%)
- 准确率: 0.0058 (0.58%)

### 初步实验分析

#### 1. 训练效率对比
- YOLO训练速度明显快于UNET（1.44分钟 vs 8-9分钟）
- YOLO对计算资源的要求相对较低
- 两个模型都能在CPU上完成训练

#### 2. 模型性能对比
- 两个模型在快速训练（3轮）条件下性能都不理想
- YOLO表现：
  - mAP50达到12.7%
  - 召回率较高(73.8%)但精确率很低(1.36%)
- UNET表现：
  - 召回率达到100%但其他指标都很低
  - 出现明显的过拟合现象

#### 3. 问题分析
1. 训练轮数不足：
   - 3轮训练无法让模型充分学习特征
   - 模型都没有达到收敛

2. 学习率设置：
   - YOLO的初始学习率(0.01)可能偏大
   - UNET的学习率调度可能需要优化

3. 数据增强不足：
   - 当前配置下关闭了部分数据增强功能
   - 可能影响模型的泛化能力

### 改进建议

1. 训练策略优化：
   - YOLO：增加训练轮数至30轮以上
   - UNET：增加训练轮数至20轮以上
   - 调整学习率策略，使用更温和的初始学习率

2. 数据增强增强：
   - 重新启用YOLO的mosaic增强
   - 为UNET添加更多的数据增强方法
   - 考虑添加更多的图像变换和噪声

3. 模型结构调整：
   - 尝试更大的YOLO模型（如yolov8s）
   - UNET可以尝试其他backbone（如efficientnet）
   - 考虑添加注意力机制

4. 训练过程监控：
   - 添加更多的验证指标
   - 实现早停机制
   - 添加模型检查点保存

### 后续计划

1. 实施上述改进建议，进行完整的训练实验
2. 添加更多的评估指标和可视化工具
3. 进行消融实验，验证各个改进的效果
4. 尝试模型集成或其他先进的检测算法

## 后续改进计划

1. YOLOv8模型优化：
   - 增加训练轮次
   - 尝试更大的模型（YOLOv8s/m）
   - 优化学习率策略
   - 增强数据增强方法

2. U-Net模型优化：
   - 增加训练轮数（50-100轮）
   - 尝试不同的学习率策略
   - 优化数据增强方法
   - 尝试其他编码器backbone（ResNet50、EfficientNet等）
   - 添加其他损失函数（如Dice Loss）

3. 功能完善：
   - 添加检测结果可视化工具
   - 支持实时视频流检测
   - 添加Web界面
   - 支持更多类型的缺陷检测

4. 工程优化：
   - 添加日志系统
   - 优化配置管理
   - 添加模型部署支持

## 引用

如果您使用了本项目，请引用以下论文：

```bibtex
@dataset{pcb-aoi,
    author = {Dongdong Li, Dan Liu, Yun Shen, Yaqi Song, Liangliang Luo},
    title = {PCB-AoI Dataset},
    year = {2023}
}
```

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。 