import argparse
import os
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import torch.nn as nn
from tqdm import tqdm
import json
from datetime import datetime

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
        project=cfg['train'].get('project', 'experiments/yolo'),
        name=cfg['train'].get('name', 'exp'),
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
    
    # 定义优化器
    optimizer_name = cfg['train']['optimizer']['name'].lower()
    if optimizer_name == 'adam':
        optimizer = Adam(
            model.parameters(),
            lr=cfg['train']['optimizer']['lr'],
            weight_decay=cfg['train']['optimizer']['weight_decay']
        )
    elif optimizer_name == 'adamw':
        optimizer = AdamW(
            model.parameters(),
            lr=cfg['train']['optimizer']['lr'],
            weight_decay=cfg['train']['optimizer']['weight_decay']
        )
    else:
        raise ValueError(f'不支持的优化器: {optimizer_name}')
    
    # 定义学习率调度器
    scheduler_name = cfg['train']['scheduler']['name']
    if scheduler_name == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=cfg['train']['scheduler']['mode'],
            factor=cfg['train']['scheduler']['factor'],
            patience=cfg['train']['scheduler']['patience'],
            min_lr=cfg['train']['scheduler']['min_lr']
        )
    elif scheduler_name == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg['train']['scheduler']['T_0'],
            T_mult=cfg['train']['scheduler']['T_mult'],
            eta_min=cfg['train']['scheduler']['eta_min']
        )
    else:
        raise ValueError(f'不支持的学习率调度器: {scheduler_name}')
    
    # 定义损失函数
    if 'pos_weight' in cfg['train']['loss']:
        pos_weight = torch.tensor([cfg['train']['loss']['pos_weight']]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(cfg['train'].get('save_dir', 'experiments/unet'))
    save_dir = save_dir / f'run_{timestamp}'
    weights_dir = save_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置文件
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(cfg, f, indent=2)
    
    # 准备记录训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
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
            
            # 更新余弦退火学习率
            if scheduler_name == 'CosineAnnealingWarmRestarts':
                scheduler.step(epoch + batch_idx / len(train_loader))
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                # 确保mask有正确的维度
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)
                val_loss += criterion(outputs, masks).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        print(f'Epoch {epoch+1}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 更新ReduceLROnPlateau学习率
        if scheduler_name == 'ReduceLROnPlateau':
            scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), weights_dir / 'best.pt')
            print(f'保存最佳模型，验证损失: {val_loss:.4f}')
        
        # 保存最后一轮模型
        if epoch == cfg['train']['epochs'] - 1:
            torch.save(model.state_dict(), weights_dir / 'last.pt')
    
    # 保存训练历史
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)


def main(args):
    # 加载配置
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
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