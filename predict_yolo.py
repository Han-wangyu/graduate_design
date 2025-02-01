import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import argparse
from ultralytics import YOLO
from utils.metrics import compute_metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='YOLO model weights path')
    parser.add_argument('--source', type=str, required=True, help='source directory with images')
    parser.add_argument('--save-dir', type=str, required=True, help='directory to save results')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--debug', action='store_true', help='save debug information')
    return parser.parse_args()

def main(args):
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if args.debug:
        debug_dir = save_dir / 'debug'
        debug_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model = YOLO(args.weights)

    # 加载图像列表
    source_dir = Path(args.source)
    image_files = sorted([f for f in source_dir.glob('*.jpeg')])
    print(f'找到{len(image_files)}张测试图像')

    # 准备评估结果
    all_predictions = []
    all_ground_truths = []

    # 处理每张图像
    for img_file in tqdm(image_files, desc='Processing images'):
        # 加载图像
        img0 = cv2.imread(str(img_file))
        if img0 is None:
            continue

        # 推理
        results = model(img0, conf=args.conf, device=args.device)[0]

        # 保存预测结果
        predictions = []
        if len(results.boxes):
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                predictions.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(box.conf),
                    'class': int(box.cls)
                })

        # 加载真实标注
        gt_file = img_file.parent.parent / 'labels' / f'{img_file.stem}.txt'
        ground_truths = []
        if gt_file.exists():
            with open(gt_file) as f:
                for line in f:
                    cls, x_center, y_center, width, height = map(float, line.strip().split())
                    # 转换为像素坐标
                    x1 = int((x_center - width/2) * img0.shape[1])
                    y1 = int((y_center - height/2) * img0.shape[0])
                    x2 = int((x_center + width/2) * img0.shape[1])
                    y2 = int((y_center + height/2) * img0.shape[0])
                    ground_truths.append({
                        'bbox': [x1, y1, x2, y2],
                        'class': int(cls)
                    })

        all_predictions.append(predictions)
        all_ground_truths.append(ground_truths)

        # 保存调试信息
        if args.debug:
            debug_img = img0.copy()
            # 绘制预测框
            for pred in predictions:
                x1, y1, x2, y2 = pred['bbox']
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 绘制真实框
            for gt in ground_truths:
                x1, y1, x2, y2 = gt['bbox']
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite(str(debug_dir / img_file.name), debug_img)

    # 计算评估指标
    print('\n计算评估指标...')
    metrics = compute_metrics(all_predictions, all_ground_truths)
    
    print('\nYOLO模型评估结果:')
    for metric, value in metrics.items():
        print(f'{metric}: {value}')

    # 保存评估结果
    with open(save_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f'\n结果保存到 {save_dir}')
    if args.debug:
        print(f'调试信息保存到 {save_dir}/debug')

if __name__ == '__main__':
    args = parse_args()
    main(args) 