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
import pandas as pd
import seaborn as sns
from ultralytics import YOLO
import segmentation_models_pytorch as smp

from models.unet.model import create_model
from utils.augmentation import get_train_transforms, get_val_transforms
from utils.data_utils import PCBDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class ExperimentManager:
    """实验管理器，负责管理多个实验的运行和结果比较"""
    
    def __init__(self, base_config_path, output_dir='runs/experiments', device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.base_config_path = base_config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载基础配置
        with open(base_config_path) as f:
            self.base_config = yaml.safe_load(f)
        
        # 实验结果存储
        self.results = []
        
    def run_experiments(self, experiment_configs):
        """运行一系列实验"""
        for config in experiment_configs:
            exp_name = f"{config['model_type']}_{config['model_variant']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"\n=== 开始实验: {exp_name} ===")
            
            # 创建实验配置
            exp_config = self._create_experiment_config(config)
            
            # 运行实验
            experiment = Experiment(exp_config, self.device, exp_name)
            metrics = experiment.run()
            
            # 记录结果
            self.results.append({
                'experiment_name': exp_name,
                'model_type': config['model_type'],
                'model_variant': config['model_variant'],
                'metrics': metrics
            })
        
        # 生成对比报告
        self._generate_comparison_report()
    
    def _create_experiment_config(self, config):
        """根据实验配置创建完整配置"""
        exp_config = self.base_config.copy()
        
        # 更新模型配置
        if config['model_type'] == 'yolo':
            exp_config['model'].update({
                'name': 'yolov8',
                'variant': config['model_variant'],
                'pretrained': True
            })
        elif config['model_type'] == 'smp':
            exp_config['model'].update({
                'name': config['model_variant'],
                'encoder_name': config.get('encoder_name', 'resnet34'),
                'encoder_weights': 'imagenet'
            })
        
        return exp_config
    
    def _generate_comparison_report(self):
        """生成实验对比报告"""
        # 创建结果DataFrame
        results_df = pd.DataFrame(self.results)
        
        # 生成对比图表
        self._plot_metrics_comparison(results_df)
        
        # 保存详细报告
        report_path = self.output_dir / 'experiment_report.md'
        self._save_markdown_report(results_df, report_path)
    
    def _plot_metrics_comparison(self, results_df):
        """生成性能对比图表"""
        plt.figure(figsize=(15, 10))
        
        # 准确率/精确率对比
        plt.subplot(2, 2, 1)
        self._plot_metric_bars(results_df, 'precision', '精确率对比')
        
        # F1分数对比
        plt.subplot(2, 2, 2)
        self._plot_metric_bars(results_df, 'f1', 'F1分数对比')
        
        # mAP50对比 (YOLO) / IoU对比 (UNET)
        plt.subplot(2, 2, 3)
        for model_type in results_df['model_type'].unique():
            model_data = results_df[results_df['model_type'] == model_type]
            if model_type == 'yolo':
                self._plot_metric_bars(model_data, 'mAP50', 'mAP50对比')
            else:
                self._plot_metric_bars(model_data, 'iou', 'IoU对比')
        
        # 训练时间对比
        plt.subplot(2, 2, 4)
        self._plot_metric_bars(results_df, 'training_time', '训练时间对比(秒)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_comparison.png')
        plt.close()
    
    def _plot_metric_bars(self, df, metric_name, title):
        """绘制单个指标的柱状图"""
        if len(df) == 0 or metric_name not in df.columns:
            return
            
        try:
            sns.barplot(
                data=df,
                x='model_variant',
                y=metric_name,
                hue='model_type'
            )
            plt.title(title)
            plt.xticks(rotation=45)
        except ValueError as e:
            print(f"Warning: Could not plot {metric_name} - {str(e)}")
    
    def _save_markdown_report(self, results_df, report_path):
        """生成Markdown格式的实验报告"""
        with open(report_path, 'w') as f:
            f.write('# PCB缺陷检测模型对比实验报告\n\n')
            f.write(f'实验时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
            
            # 实验配置
            f.write('## 实验配置\n\n')
            f.write(f'- 设备: {self.device}\n')
            f.write(f'- 基础配置文件: {self.base_config_path}\n\n')
            
            # 模型性能对比
            f.write('## 模型性能对比\n\n')
            f.write('### 性能指标\n\n')
            
            # 分别展示YOLO和其他模型的指标
            for model_type in results_df['model_type'].unique():
                f.write(f'\n#### {model_type.upper()} 模型\n\n')
                model_results = results_df[results_df['model_type'] == model_type]
                f.write(model_results.to_markdown(index=False))
            
            # 可视化结果
            f.write('\n\n### 可视化对比\n\n')
            f.write('![性能对比](metrics_comparison.png)\n\n')
            
            # 结论和建议
            f.write('## 结论和建议\n\n')
            # 对于YOLO模型，使用mAP50作为主要指标
            yolo_results = results_df[results_df['model_type'] == 'yolo']
            if not yolo_results.empty and 'mAP50' in yolo_results.columns:
                best_yolo = yolo_results.loc[yolo_results['mAP50'].idxmax()]
                f.write(f'- 最佳YOLO模型: {best_yolo["model_variant"]}\n')
                f.write(f'- mAP50: {best_yolo["mAP50"]:.4f}\n')
            
            # 对于其他模型，使用F1分数作为主要指标
            other_results = results_df[results_df['model_type'] != 'yolo']
            if not other_results.empty and 'f1' in other_results.columns:
                best_other = other_results.loc[other_results['f1'].idxmax()]
                f.write(f'- 最佳分割模型: {best_other["model_variant"]}\n')
                f.write(f'- F1分数: {best_other["f1"]:.4f}\n')
                if 'iou' in best_other:
                    f.write(f'- IoU: {best_other["iou"]:.4f}\n')

class Experiment:
    """单个实验的实现"""
    
    def __init__(self, config, device='cpu', exp_name=None):
        self.device = device
        self.config = config
        self.exp_name = exp_name or datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建输出目录
        self.output_dir = Path(f'runs/experiments/{self.exp_name}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)
    
    def run(self):
        """运行实验"""
        if self.config['model']['name'] == 'yolov8':
            return self.train_yolo()
        else:
            return self.train_smp()
    
    def train_yolo(self):
        """训练YOLO模型"""
        print(f"\n=== 开始YOLO训练: {self.config['model']['variant']} ===")
        start_time = time.time()
        
        # 获取当前工作目录的绝对路径
        current_dir = os.getcwd()
        
        # 准备数据集配置
        data_yaml = {
            'path': os.path.join(current_dir, 'data/dataset'),
            'train': os.path.join(current_dir, 'data/dataset/train_data/images'),
            'val': os.path.join(current_dir, 'data/dataset/test_data/images'),
            'names': ['defect'],
            'nc': self.config['model']['num_classes']
        }
        
        data_yaml_path = self.output_dir / 'data.yaml'
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
        
        # 创建模型
        model = YOLO(f"{self.config['model']['variant']}.yaml")
        if self.config['model']['pretrained']:
            model = YOLO(f"{self.config['model']['variant']}.pt")
        
        # 训练模型
        results = model.train(
            data=str(data_yaml_path),
            epochs=self.config['train']['epochs'],
            imgsz=self.config['data']['img_size'],
            batch=self.config['data']['batch_size'],
            device=self.device,
            workers=self.config['data']['workers'],
            pretrained=self.config['model']['pretrained'],
            lr0=self.config['train']['lr0'],
            lrf=self.config['train']['lrf'],
            momentum=self.config['train']['momentum'],
            weight_decay=self.config['train']['weight_decay'],
            warmup_epochs=self.config['train']['warmup_epochs'],
            warmup_momentum=self.config['train']['warmup_momentum'],
            warmup_bias_lr=self.config['train']['warmup_bias_lr'],
            val=True,
            save=True,
            project=str(self.output_dir),
            name='yolo',
            exist_ok=True
        )
        
        training_time = time.time() - start_time
        
        # 提取指标
        precision = float(results.results_dict['metrics/precision(B)'])
        recall = float(results.results_dict['metrics/recall(B)'])
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'mAP50': float(results.results_dict['metrics/mAP50(B)']),
            'mAP50-95': float(results.results_dict['metrics/mAP50-95(B)']),
            'f1': 2 * precision * recall / (precision + recall + 1e-16),  # 添加小量防止除零
            'training_time': training_time
        }
        
        return metrics
    
    def train_smp(self):
        """训练SMP模型"""
        print(f"\n=== 开始SMP训练: {self.config['model']['name']} ===")
        start_time = time.time()
        
        # 创建模型
        model = self._create_smp_model()
        model = model.to(self.device)
        
        # 准备数据加载器
        train_loader, val_loader = self._create_dataloaders()
        
        # 训练模型
        metrics = self._train_smp_model(model, train_loader, val_loader)
        metrics['training_time'] = time.time() - start_time
        
        return metrics
    
    def _create_smp_model(self):
        """创建SMP模型"""
        model_params = {
            'encoder_name': self.config['model']['encoder_name'],
            'encoder_weights': self.config['model']['encoder_weights'],
            'in_channels': self.config['model']['in_channels'],
            'classes': self.config['model']['classes']
        }
        
        if self.config['model']['name'] == 'unet':
            return smp.Unet(**model_params)
        elif self.config['model']['name'] == 'deeplabv3plus':
            return smp.DeepLabV3Plus(**model_params)
        elif self.config['model']['name'] == 'fpn':
            return smp.FPN(**model_params)
        elif self.config['model']['name'] == 'pspnet':
            return smp.PSPNet(**model_params)
        elif self.config['model']['name'] == 'manet':
            return smp.MAnet(**model_params)
        else:
            raise ValueError(f"不支持的模型类型: {self.config['model']['name']}")
    
    def _create_dataloaders(self):
        """创建数据加载器"""
        train_transform = get_train_transforms(self.config)
        val_transform = get_val_transforms(self.config)
        
        train_dataset = PCBDataset(
            img_dir=os.path.join(self.config['data']['train_path'], 'images'),
            mask_dir=os.path.join(self.config['data']['train_path'], 'masks'),
            transform=train_transform
        )
        
        val_dataset = PCBDataset(
            img_dir=os.path.join(self.config['data']['val_path'], 'images'),
            mask_dir=os.path.join(self.config['data']['val_path'], 'masks'),
            transform=val_transform
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['workers']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['workers']
        )
        
        return train_loader, val_loader
    
    def _train_smp_model(self, model, train_loader, val_loader):
        """训练SMP模型"""
        optimizer = Adam(
            model.parameters(),
            lr=self.config['train']['optimizer']['lr'],
            weight_decay=self.config['train']['optimizer']['weight_decay']
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=self.config['train']['scheduler']['mode'],
            factor=self.config['train']['scheduler']['factor'],
            patience=self.config['train']['scheduler']['patience'],
            min_lr=self.config['train']['scheduler']['min_lr']
        )
        
        criterion = nn.BCEWithLogitsLoss()
        
        best_loss = float('inf')
        metrics_history = []
        
        for epoch in range(self.config['train']['epochs']):
            # 训练阶段
            model.train()
            train_loss = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["train"]["epochs"]}')
            for images, masks in pbar:
                images, masks = images.to(self.device), masks.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
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
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            epoch_metrics.update({k: np.mean(v) for k, v in val_metrics.items()})
            
            metrics_history.append(epoch_metrics)
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                save_path = self.output_dir / 'weights'
                save_path.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path / 'best.pt')
        
        # 返回最后一轮的指标
        return metrics_history[-1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-config', type=str, default='configs/base_config.yaml',
                      help='基础配置文件路径')
    parser.add_argument('--device', type=str, default='cpu',
                      help='训练设备')
    args = parser.parse_args()
    
    # 创建实验管理器
    manager = ExperimentManager(args.base_config, device=args.device)
    
    # 定义要运行的实验（精简版本）
    experiment_configs = [
        # YOLO实验 - 使用最轻量级的模型
        {'model_type': 'yolo', 'model_variant': 'yolov8n'},
        
        # SMP实验 - 使用基础的UNET
        {'model_type': 'smp', 'model_variant': 'unet', 'encoder_name': 'resnet34'},
    ]
    
    # 运行实验
    manager.run_experiments(experiment_configs)

if __name__ == '__main__':
    main() 