import argparse
import os
import yaml
import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import time
import json
from ultralytics import YOLO

from models.unet.model import create_model
from utils.augmentation import get_train_transforms, get_val_transforms
from utils.data_utils import PCBDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class Experiment:
    def __init__(self, config_path, device='cpu', exp_name=None):
        """初始化实验"""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config_path = config_path
        self.exp_name = exp_name or datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 获取当前工作目录的绝对路径
        self.current_dir = os.getcwd()
        
        # 加载配置
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        
        # 创建输出目录
        self.output_dir = Path(f'runs/experiments/{self.exp_name}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.cfg, f)
    
    def train_yolo(self):
        """训练YOLO模型"""
        print("\n=== 开始YOLO训练 ===")
        start_time = time.time()
        
        # 准备数据集配置
        data_yaml = {
            'path': os.path.join(self.current_dir, 'data/dataset'),
            'train': os.path.join(self.current_dir, 'data/dataset/train_data/images'),
            'val': os.path.join(self.current_dir, 'data/dataset/test_data/images'),
            'names': ['defect'],
            'nc': self.cfg['model']['num_classes']
        }
        
        data_yaml_path = self.output_dir / 'data.yaml'
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
        
        # 创建模型
        model = YOLO(f"{self.cfg['model']['variant']}.yaml")
        if self.cfg['model']['pretrained']:
            model = YOLO(f"{self.cfg['model']['variant']}.pt")
        
        # 训练模型
        results = model.train(
            data=str(data_yaml_path),
            epochs=self.cfg['train']['epochs'],
            imgsz=self.cfg['data']['img_size'],
            batch=self.cfg['data']['batch_size'],
            device=self.device,
            workers=self.cfg['data']['workers'],
            pretrained=self.cfg['model']['pretrained'],
            lr0=self.cfg['train']['lr0'],
            lrf=self.cfg['train']['lrf'],
            momentum=self.cfg['train']['momentum'],
            weight_decay=self.cfg['train']['weight_decay'],
            warmup_epochs=self.cfg['train']['warmup_epochs'],
            warmup_momentum=self.cfg['train']['warmup_momentum'],
            warmup_bias_lr=self.cfg['train']['warmup_bias_lr'],
            val=True,
            save=True,
            project=str(self.output_dir),
            name='yolo',
            exist_ok=True
        )
        
        training_time = time.time() - start_time
        
        # 从验证结果中提取最终指标
        final_metrics = {
            'precision': float(results.results_dict['metrics/precision(B)']),
            'recall': float(results.results_dict['metrics/recall(B)']),
            'mAP50': float(results.results_dict['metrics/mAP50(B)']),
            'mAP50-95': float(results.results_dict['metrics/mAP50-95(B)']),
            'fitness': float(results.fitness)
        }
        
        # 记录训练结果
        metrics = {
            'model': 'yolo',
            'training_time': training_time,
            'final_metrics': final_metrics,
            'training_config': {
                'epochs': self.cfg['train']['epochs'],
                'batch_size': self.cfg['data']['batch_size'],
                'image_size': self.cfg['data']['img_size'],
                'device': str(self.device)
            }
        }
        
        with open(self.output_dir / 'yolo_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return metrics
    
    def train_unet(self):
        """训练UNET模型"""
        print("\n=== 开始UNET训练 ===")
        start_time = time.time()
        
        # 创建模型
        model = create_model(self.cfg).to(self.device)
        
        # 准备数据加载器
        train_transform = get_train_transforms(self.cfg)
        val_transform = get_val_transforms(self.cfg)
        
        train_dataset = PCBDataset(
            img_dir=os.path.join(self.cfg['data']['train_path'], 'images'),
            mask_dir=os.path.join(self.cfg['data']['train_path'], 'masks'),
            transform=train_transform
        )
        
        val_dataset = PCBDataset(
            img_dir=os.path.join(self.cfg['data']['val_path'], 'images'),
            mask_dir=os.path.join(self.cfg['data']['val_path'], 'masks'),
            transform=val_transform
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg['data']['batch_size'],
            shuffle=True,
            num_workers=self.cfg['data']['workers']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg['data']['batch_size'],
            shuffle=False,
            num_workers=self.cfg['data']['workers']
        )
        
        # 定义优化器和损失函数
        optimizer = Adam(
            model.parameters(),
            lr=self.cfg['train']['optimizer']['lr'],
            weight_decay=self.cfg['train']['optimizer']['weight_decay']
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=self.cfg['train']['scheduler']['mode'],
            factor=self.cfg['train']['scheduler']['factor'],
            patience=self.cfg['train']['scheduler']['patience'],
            min_lr=self.cfg['train']['scheduler']['min_lr']
        )
        
        criterion = nn.BCEWithLogitsLoss()
        
        # 训练循环
        best_loss = float('inf')
        metrics_history = []
        
        for epoch in range(self.cfg['train']['epochs']):
            # 训练阶段
            model.train()
            train_loss = 0
            batch_times = []
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.cfg["train"]["epochs"]}')
            for batch_idx, (images, masks) in enumerate(pbar):
                batch_start = time.time()
                
                images, masks = images.to(self.device), masks.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                batch_times.append(time.time() - batch_start)
                pbar.set_postfix({'loss': loss.item()})
            
            # 验证阶段
            model.eval()
            val_loss = 0
            val_metrics = {'iou': [], 'precision': [], 'recall': [], 'f1': [], 'accuracy': []}
            
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(self.device), masks.to(self.device)
                    outputs = model(images)
                    val_loss += criterion(outputs, masks).item()
                    
                    # 计算指标
                    pred = (torch.sigmoid(outputs) > 0.5).float()
                    pred = pred.cpu().numpy()
                    masks = masks.cpu().numpy()
                    
                    for p, m in zip(pred, masks):
                        p = p.flatten()
                        m = m.flatten()
                        val_metrics['precision'].append(precision_score(m, p, zero_division=0))
                        val_metrics['recall'].append(recall_score(m, p, zero_division=0))
                        val_metrics['f1'].append(f1_score(m, p, zero_division=0))
                        val_metrics['accuracy'].append(accuracy_score(m, p))
                        
                        intersection = np.logical_and(p, m).sum()
                        union = np.logical_or(p, m).sum()
                        val_metrics['iou'].append(intersection / (union + 1e-6))
            
            # 计算平均指标
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'batch_time': np.mean(batch_times),
                'metrics': {
                    k: np.mean(v) for k, v in val_metrics.items()
                }
            }
            
            metrics_history.append(epoch_metrics)
            
            print(f'\nEpoch {epoch+1}:')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print('Validation Metrics:')
            for k, v in epoch_metrics['metrics'].items():
                print(f'{k}: {v:.4f}')
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                save_path = self.output_dir / 'unet/weights'
                save_path.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path / 'best.pt')
        
        training_time = time.time() - start_time
        
        # 记录训练结果
        metrics = {
            'model': 'unet',
            'training_time': training_time,
            'metrics_history': metrics_history
        }
        
        with open(self.output_dir / 'unet_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return metrics
    
    def run_comparison(self):
        """运行对比实验"""
        print("开始对比实验...")
        
        # 运行两个模型的训练
        yolo_metrics = self.train_yolo()
        unet_metrics = self.train_unet()
        
        # 生成对比报告
        report = {
            'experiment_name': self.exp_name,
            'device': str(self.device),
            'yolo': yolo_metrics,
            'unet': unet_metrics
        }
        
        # 保存报告
        with open(self.output_dir / 'comparison_report.json', 'w') as f:
            json.dump(report, f, indent=4)
        
        # 生成可视化对比
        self.visualize_comparison(report)
        
        return report
    
    def visualize_comparison(self, report):
        """生成对比可视化"""
        # 创建图表
        plt.figure(figsize=(15, 10))
        
        # 训练时间对比
        plt.subplot(2, 2, 1)
        times = [report['yolo']['training_time'], report['unet']['training_time']]
        plt.bar(['YOLO', 'U-Net'], times)
        plt.title('训练时间对比 (秒)')
        
        # 最终性能指标对比
        plt.subplot(2, 2, 2)
        yolo_metrics = report['yolo']['final_metrics']
        unet_metrics = report['unet']['metrics_history'][-1]['metrics']
        
        metrics_to_compare = ['precision', 'recall', 'f1', 'accuracy']
        x = np.arange(len(metrics_to_compare))
        width = 0.35
        
        plt.bar(x - width/2, [yolo_metrics.get(m, 0) for m in metrics_to_compare], width, label='YOLO')
        plt.bar(x + width/2, [unet_metrics.get(m, 0) for m in metrics_to_compare], width, label='U-Net')
        plt.xticks(x, metrics_to_compare)
        plt.title('性能指标对比')
        plt.legend()
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comparison_plots.png')
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-config', type=str, default='configs/yolov8.yaml',
                      help='YOLO配置文件路径')
    parser.add_argument('--unet-config', type=str, default='configs/unet.yaml',
                      help='UNET配置文件路径')
    parser.add_argument('--device', type=str, default='cpu',
                      help='训练设备')
    parser.add_argument('--exp-name', type=str, default=None,
                      help='实验名称')
    args = parser.parse_args()
    
    # 运行YOLO实验
    yolo_exp = Experiment(args.yolo_config, args.device, f"{args.exp_name}_yolo" if args.exp_name else None)
    yolo_metrics = yolo_exp.train_yolo()
    
    # 运行UNET实验
    unet_exp = Experiment(args.unet_config, args.device, f"{args.exp_name}_unet" if args.exp_name else None)
    unet_metrics = unet_exp.train_unet()
    
    # 生成对比报告
    report = {
        'experiment_name': args.exp_name or datetime.now().strftime('%Y%m%d_%H%M%S'),
        'device': args.device,
        'yolo': yolo_metrics,
        'unet': unet_metrics
    }
    
    # 保存报告
    output_dir = Path('runs/experiments')
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f'{report["experiment_name"]}_comparison.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    print("\n=== 实验完成 ===")
    print(f"结果已保存到: {output_dir}")

if __name__ == '__main__':
    main() 