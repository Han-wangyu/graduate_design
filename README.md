# PCB表面缺陷检测系统

基于深度学习的PCB板焊接缺陷检测系统，支持多种先进的检测算法。本项目使用PCB-AoI数据集进行训练和评估，支持YOLOv8和U-Net等多种模型的训练和对比实验，并提供模型融合功能以提高检测准确度。

## 功能特点

- 支持多种深度学习模型
  - 目标检测：YOLOv8（n/s/m）
  - 语义分割：U-Net、DeepLabV3+、FPN、PSPNet、MANet等
- 完整的实验管理框架
  - 统一的配置管理
  - 自动化的实验运行
  - 实验结果可视化
  - 模型性能对比
- 模型融合决策
  - YOLO和UNET模型融合
  - 基于置信度的加权决策
  - 区域验证和互补检测
  - 自适应阈值调整
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
│   ├── unet/        # U-Net模型实现
│   └── fusion.py    # 模型融合实现
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
├── predict_fusion.py # 融合模型预测脚本
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

#### 单模型预测
```bash
# YOLOv8预测
python3 predict.py --weights runs/train/exp/weights/best.pt --source data/dataset/test_data/images

# U-Net预测
python3 predict_unet.py --weights runs/train/exp/weights/best.pt --source data/dataset/test_data/images
```

#### 融合模型预测
```bash
# 使用融合模型预测
python3 predict_fusion.py \
    --yolo-weights runs/train/yolo/weights/best.pt \
    --unet-weights runs/train/unet/weights/best.pt \
    --source data/dataset/test_data/images \
    --device cpu \
    --yolo-conf 0.25 \
    --unet-conf 0.5
```

## 模型融合策略

### 1. 基本原理
- YOLO提供精确的目标定位和分类
- UNET提供像素级的分割信息
- 融合决策结合两个模型的优势

### 2. 融合方法
- 区域验证：使用YOLO定位区域，UNET验证区域内的缺陷
- 置信度加权：根据各模型的置信度进行加权决策
- 互补检测：利用两个模型的互补性提高检测准确度
- 自适应阈值：根据两个模型的预测一致性动态调整阈值

### 3. 参数调整
- yolo-conf：YOLO模型的置信度阈值
- unet-conf：UNET模型的置信度阈值
- iou-thres：区域重叠的IoU阈值
- min-area：最小缺陷区域面积

## 实验结果

### 1. 单模型性能

#### YOLOv8
- 置信度阈值：0.7
- Precision: 0.58
- Recall: 0.62
- F1 Score: 0.60
- IoU Score: 0.42

#### U-Net (ResNet34)
- 置信度阈值：0.8
- Precision: 0.55
- Recall: 0.58
- F1 Score: 0.56
- IoU Score: 0.40

### 2. 融合模型性能

最新实验结果（2024年2月）：
- YOLO置信度阈值：0.7
- UNET置信度阈值：0.8
- Precision: 0.616
- Recall: 0.605
- F1 Score: 0.610
- IoU Score: 0.439
- True Positives: 61,241
- False Positives: 38,162
- False Negatives: 40,047

### 3. 运行命令示例

```bash
# 运行融合模型预测（最佳参数配置）
python3 predict_fusion_experiment.py \
    --yolo-weights runs/train/exp/weights/best.pt \
    --unet-weights experiments/unet/run_20250201_214043/weights/best.pt \
    --yolo-conf 0.7 \
    --unet-conf 0.8 \
    --source data/dataset/test_data/images \
    --debug
```

### 4. 关键发现

1. 融合模型相比单模型均有提升：
   - F1分数提升至0.61（相比YOLO提升1%，相比UNET提升5%）
   - 精确率和召回率达到更好的平衡

2. 置信度阈值的影响：
   - YOLO：较高的置信度（0.7）有助于提高精确率
   - UNET：较高的置信度（0.8）可以减少误检
   
3. 模型互补性：
   - YOLO擅长定位大面积缺陷
   - UNET在细节和边缘检测上表现更好
   - 融合后能够同时保持两个模型的优势

## 后续改进计划

1. 模型优化：
   - 增加训练轮数（50-100轮）
   - 尝试更大的模型（YOLOv8s/m）
   - 优化学习率策略
   - 增强数据增强方法

2. 融合策略优化：
   - 添加更多融合策略选项
   - 实现自动参数优化
   - 添加更多评估指标
   - 支持在线模型选择

3. 功能完善：
   - 添加TensorBoard支持
   - 添加模型集成功能
   - 支持更多的评估指标
   - 添加交互式可视化界面

4. 工程优化：
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