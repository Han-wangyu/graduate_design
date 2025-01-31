import argparse
import os
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from utils.augmentation import get_train_transforms, get_val_transforms
from utils.data_utils import create_dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/yolov8.yaml',
                      help='配置文件路径')
    parser.add_argument('--device', type=str, default='cuda',
                      help='训练设备')
    return parser.parse_args()


def train_yolov8(cfg):
    """训练YOLOv8模型

    Args:
        cfg: 配置信息
    """
    # 获取当前工作目录的绝对路径
    current_dir = os.getcwd()
    
    # 创建模型
    model = YOLO(f"{cfg['model']['variant']}.yaml")
    
    if cfg['model']['pretrained']:
        model = YOLO(f"{cfg['model']['variant']}.pt")
    
    # 准备数据集配置
    data_yaml = {
        'path': os.path.join(current_dir, 'data/dataset'),
        'train': os.path.join(current_dir, 'data/dataset/train_data/images'),
        'val': os.path.join(current_dir, 'data/dataset/test_data/images'),
        'names': ['defect'],
        'nc': cfg['model']['num_classes']
    }
    
    # 保存数据集配置
    with open('data.yaml', 'w') as f:
        yaml.dump(data_yaml, f)
    
    # 训练模型
    model.train(
        data='data.yaml',
        epochs=cfg['train']['epochs'],
        imgsz=cfg['data']['img_size'],
        batch=cfg['data']['batch_size'],
        device=args.device,
        workers=cfg['data']['workers'],
        pretrained=cfg['model']['pretrained'],
        lr0=cfg['train']['lr0'],
        lrf=cfg['train']['lrf'],
        momentum=cfg['train']['momentum'],
        weight_decay=cfg['train']['weight_decay'],
        warmup_epochs=cfg['train']['warmup_epochs'],
        warmup_momentum=cfg['train']['warmup_momentum'],
        warmup_bias_lr=cfg['train']['warmup_bias_lr'],
        val=True,
        save=True,
        save_period=10,
        project='runs/train',
        name='exp',
        exist_ok=True
    )


def main(args):
    # 加载配置
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 创建输出目录
    save_dir = Path('runs/train/exp')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 根据模型类型选择训练函数
    if cfg['model']['name'] == 'yolov8':
        train_yolov8(cfg)
    else:
        raise NotImplementedError(f"Model {cfg['model']['name']} not implemented")


if __name__ == '__main__':
    args = parse_args()
    main(args) 