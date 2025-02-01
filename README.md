# PCB表面缺陷检测系统

基于深度学习的PCB板焊接缺陷检测系统，支持多种先进的检测算法。本项目使用PCB-AoI数据集进行训练和评估，支持YOLOv8和U-Net等多种模型的训练和对比实验。

## 功能特点

- 支持多种深度学习模型
  - 目标检测：YOLOv8（n/s/m）
  - 语义分割：U-Net、DeepLabV3+、FPN、PSPNet、MANet等
- 完整的实验管理框架
  - 统一的配置管理
  - 自动化的实验运行
  - 实验结果可视化
  - 模型性能对比
- 支持CPU和GPU训练
- 提供完整的训练、验证和推理流程

## 项目结构

```
├── configs/           # 配置文件目录
│   ├── base_config.yaml   # 基础配置
│   ├── yolov8.yaml       # YOLOv8模型配置
│   └── unet.yaml         # U-Net模型配置
├── data/             # 数据集目录
│   ├── dataset/      # PCB-AoI数据集
│   └── pcb-aoi.md   # 数据集说明文档
├── models/           # 模型定义
│   └── unet/        # U-Net模型实现
├── tools/            # 工具脚本
│   ├── convert_voc_to_yolo.py  # VOC转YOLO格式工具
│   └── convert_voc_to_mask.py  # VOC转分割掩码工具
├── utils/            # 通用工具函数
│   ├── augmentation.py  # 数据增强
│   └── data_utils.py    # 数据处理
├── experiment.py     # 实验管理主程序
├── train.py         # 训练脚本
├── evaluate.py      # 评估脚本
├── predict.py       # YOLO预测脚本
├── predict_unet.py  # UNET预测脚本
└── requirements.txt  # 依赖包
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

### 1. 运行实验

使用实验管理框架运行对比实验：
```bash
# 使用CPU运行基础实验（YOLOv8n和U-Net）
python3 experiment.py --base-config configs/base_config.yaml --device cpu

# 使用GPU运行实验（如果可用）
python3 experiment.py --base-config configs/base_config.yaml --device cuda
```

### 2. 单独训练模型

```bash
# 训练YOLOv8
python3 train.py --config configs/yolov8.yaml --device cpu

# 训练U-Net
python3 train.py --config configs/unet.yaml --device cpu
```

### 3. 预测

```bash
# YOLOv8预测
python3 predict.py --weights runs/train/exp/weights/best.pt --source data/dataset/test_data/images

# U-Net预测
python3 predict_unet.py --weights runs/train/exp/weights/best.pt --source data/dataset/test_data/images
```

## 实验结果

### 快速实验结果（3轮训练）

1. YOLOv8n:
   - mAP50: 0.127
   - Precision: 0.014
   - Recall: 0.738
   - F1 Score: 0.027
   - 训练时间: 103秒

2. U-Net (ResNet34):
   - IoU: 0.006
   - Precision: 0.006
   - Recall: 1.000
   - F1 Score: 0.011
   - 训练时间: 634秒

## 后续改进计划

1. 模型优化：
   - 增加训练轮数（50-100轮）
   - 尝试更大的模型（YOLOv8s/m）
   - 优化学习率策略
   - 增强数据增强方法

2. 功能完善：
   - 添加TensorBoard支持
   - 添加模型集成功能
   - 支持更多的评估指标
   - 添加交互式可视化界面

3. 工程优化：
   - 添加分布式训练支持
   - 优化数据加载性能
   - 添加模型压缩功能
   - 支持模型导出（ONNX等）

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