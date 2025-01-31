import argparse
import os
from pathlib import Path

import cv2
import torch
import yaml
from ultralytics import YOLO
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True,
                      help='模型权重路径')
    parser.add_argument('--source', type=str, required=True,
                      help='输入图像路径或目录')
    parser.add_argument('--config', type=str, default='configs/yolov8.yaml',
                      help='配置文件路径')
    parser.add_argument('--device', type=str, default='cuda',
                      help='推理设备')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                      help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                      help='NMS IoU阈值')
    parser.add_argument('--save-txt', action='store_true',
                      help='保存文本结果')
    return parser.parse_args()


def predict_yolov8(model, source, cfg, device, conf_thres, iou_thres, save_txt=False):
    """使用YOLOv8模型进行预测

    Args:
        model: YOLOv8模型
        source: 输入图像路径或目录
        cfg: 配置信息
        device: 设备
        conf_thres: 置信度阈值
        iou_thres: NMS IoU阈值
        save_txt: 是否保存文本结果
    """
    # 创建输出目录
    save_dir = Path('runs/predict/exp')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 进行预测
    results = model.predict(
        source=source,
        conf=conf_thres,
        iou=iou_thres,
        imgsz=cfg['data']['img_size'],
        device=device,
        save=True,
        save_txt=save_txt,
        save_conf=True,
        project='runs/predict',
        name='exp',
        exist_ok=True
    )
    
    return results


def main(args):
    # 加载配置
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载模型并进行预测
    if cfg['model']['name'] == 'yolov8':
        model = YOLO(args.weights)
        results = predict_yolov8(
            model=model,
            source=args.source,
            cfg=cfg,
            device=device,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            save_txt=args.save_txt
        )
        
        # 打印预测结果统计
        print('\nPrediction Results:')
        print(f'预测完成，结果保存在: {Path("runs/predict/exp")}')
        if args.save_txt:
            print(f'检测结果文本保存在: {Path("runs/predict/exp/labels")}')
    else:
        raise NotImplementedError(f"Model {cfg['model']['name']} not implemented")


if __name__ == '__main__':
    args = parse_args()
    main(args) 