# PCBA表面缺陷检测系统

基于深度学习的PCB板焊接缺陷检测系统，支持多种先进的检测算法。本项目使用PCB-AoI数据集进行训练和评估，目前已完成YOLOv8模型的基础训练和测试。

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
├── tools/            # 工具脚本
│   └── convert_voc_to_yolo.py  # VOC转YOLO格式工具
├── utils/            # 通用工具函数
│   ├── augmentation.py  # 数据增强
│   └── data_utils.py    # 数据处理
├── train.py         # 训练脚本
├── evaluate.py      # 评估脚本
├── predict.py       # 预测脚本
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
- ultralytics >= 8.0.0
- albumentations >= 1.3.0
- opencv-python >= 4.7.0

## 数据准备

1. 下载PCB-AoI数据集：
   - [Kaggle链接](https://www.kaggle.com/datasets/kubeedgeianvs/pcb-aoi)
   - [Huawei OBS链接](https://kubeedge.obs.cn-north-1.myhuaweicloud.com:443/ianvs/pcb-aoi/dataset.zip)

2. 解压数据集到 `data/dataset` 目录

3. 转换数据格式：
```bash
python3 tools/convert_voc_to_yolo.py
```

## 使用方法

### 1. 训练模型

```bash
python3 train.py --config configs/yolov8.yaml --device cpu  # 使用CPU训练
# 或
python3 train.py --config configs/yolov8.yaml --device cuda  # 使用GPU训练
```

### 2. 预测

```bash
python3 predict.py --weights runs/train/exp/weights/best.pt --source data/dataset/test_data/images --device cpu
```

## 初步实验结果

### YOLOv8n模型（10轮训练）

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

## 常见问题解决

1. 数据集路径问题
   - 问题：YOLOv8训练时找不到数据集
   - 解决：使用绝对路径或相对于工作目录的路径配置data.yaml

2. 环境配置问题
   - 问题：在不同shell环境下Python命令可能不同
   - 解决：根据实际环境使用python/python3/pip/pip3

## 后续改进计划

1. 模型性能优化：
   - 增加训练轮次
   - 尝试更大的模型（YOLOv8s/m）
   - 优化学习率策略
   - 增强数据增强方法

2. 功能完善：
   - 添加检测结果可视化工具
   - 支持实时视频流检测
   - 添加Web界面
   - 支持更多类型的缺陷检测

3. 工程优化：
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