import argparse
import os
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from tqdm import tqdm

from utils.augmentation import get_train_transforms, get_val_transforms
from utils.data_utils import create_dataloader, PCBDataset
from models.unet.model import create_model


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


def train_unet(cfg, device):
    """训练UNET模型

    Args:
        cfg: 配置信息
        device: 训练设备
    """
    # 创建模型
    model = create_model(cfg).to(device)
    
    # 创建数据集和数据加载器
    train_transform = get_train_transforms(cfg)
    val_transform = get_val_transforms(cfg)
    
    train_dataset = PCBDataset(
        img_dir=os.path.join(cfg['data']['train_path'], 'images'),
        mask_dir=os.path.join(cfg['data']['train_path'], 'masks'),
        transform=train_transform
    )
    
    val_dataset = PCBDataset(
        img_dir=os.path.join(cfg['data']['val_path'], 'images'),
        mask_dir=os.path.join(cfg['data']['val_path'], 'masks'),
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['data']['batch_size'],
        shuffle=True,
        num_workers=cfg['data']['workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['data']['batch_size'],
        shuffle=False,
        num_workers=cfg['data']['workers']
    )
    
    # 定义优化器和损失函数
    optimizer = Adam(
        model.parameters(),
        lr=cfg['train']['optimizer']['lr'],
        weight_decay=cfg['train']['optimizer']['weight_decay']
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=cfg['train']['scheduler']['mode'],
        factor=cfg['train']['scheduler']['factor'],
        patience=cfg['train']['scheduler']['patience'],
        min_lr=cfg['train']['scheduler']['min_lr']
    )
    
    criterion = nn.BCEWithLogitsLoss()
    
    # 训练循环
    best_loss = float('inf')
    for epoch in range(cfg['train']['epochs']):
        model.train()
        train_loss = 0
        
        # 训练一个epoch
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg["train"]["epochs"]}')
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            save_path = Path('runs/train/exp/weights')
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path / 'best.pt')


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
    elif cfg['model']['name'] == 'unet':
        train_unet(cfg, device)
    else:
        raise NotImplementedError(f"Model {cfg['model']['name']} not implemented")


if __name__ == '__main__':
    args = parse_args()
    main(args) 