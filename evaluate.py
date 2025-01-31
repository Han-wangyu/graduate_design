import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO
from utils.data_utils import create_dataloader
from utils.augmentation import get_val_transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True,
                      help='模型权重路径')
    parser.add_argument('--config', type=str, default='configs/yolov8.yaml',
                      help='配置文件路径')
    parser.add_argument('--device', type=str, default='cuda',
                      help='评估设备')
    return parser.parse_args()


def evaluate_yolov8(model, cfg, device):
    """评估YOLOv8模型

    Args:
        model: YOLOv8模型
        cfg: 配置信息
        device: 设备
    """
    # 准备验证数据集
    val_loader = create_dataloader(
        data_dir=cfg['data']['val_path'],
        batch_size=cfg['data']['batch_size'],
        img_size=cfg['data']['img_size'],
        transform=get_val_transforms(cfg['data']['img_size']),
        is_train=False,
        workers=cfg['data']['workers']
    )
    
    # 在验证集上评估
    results = model.val(
        data='data.yaml',
        batch=cfg['data']['batch_size'],
        imgsz=cfg['data']['img_size'],
        device=device,
        conf=cfg['val']['conf_thres'],
        iou=cfg['val']['iou_thres'],
        max_det=cfg['val']['max_det'],
        save_json=True,
        save_hybrid=True,
        plots=True
    )
    
    return results


def main(args):
    # 加载配置
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载模型
    if cfg['model']['name'] == 'yolov8':
        model = YOLO(args.weights)
        results = evaluate_yolov8(model, cfg, device)
        
        # 打印评估结果
        print('\nValidation Results:')
        print(f"mAP@0.5: {results.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {results.box.map:.4f}")
        print(f"Precision: {results.box.mp:.4f}")
        print(f"Recall: {results.box.mr:.4f}")
    else:
        raise NotImplementedError(f"Model {cfg['model']['name']} not implemented")


if __name__ == '__main__':
    args = parse_args()
    main(args) 