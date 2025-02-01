import argparse
import os
import yaml
import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import json

from models.unet.model import create_model
from models.fusion import DecisionFusion
from utils.augmentation import get_val_transforms

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', type=str, default='runs/train/exp/weights/best.pt',
                      help='YOLO模型权重路径')
    parser.add_argument('--unet-weights', type=str, default='runs/train/exp/weights/best.pt',
                      help='UNET模型权重路径')
    parser.add_argument('--unet-config', type=str, default='configs/experiment_unet.yaml',
                      help='UNET配置文件路径')
    parser.add_argument('--source', type=str, default='data/dataset/test_data/images',
                      help='测试图像目录')
    parser.add_argument('--save-dir', type=str, default='experiments/fusion',
                      help='结果保存目录')
    parser.add_argument('--device', type=str, default='cpu',
                      help='预测设备')
    parser.add_argument('--yolo-conf', type=float, default=0.25,
                      help='YOLO置信度阈值')
    parser.add_argument('--unet-conf', type=float, default=0.5,
                      help='UNET置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.5,
                      help='IoU阈值')
    parser.add_argument('--min-area', type=int, default=100,
                      help='最小缺陷区域面积')
    parser.add_argument('--debug', action='store_true',
                      help='是否保存调试信息和中间结果')
    return parser.parse_args()

def save_debug_image(save_dir: Path, name: str, image: np.ndarray, mask: np.ndarray = None):
    """保存调试图像

    Args:
        save_dir: 保存目录
        name: 图像名称
        image: 原始图像
        mask: 掩码（可选）
    """
    if mask is not None:
        # 确保掩码尺寸与图像一致
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]))
        # 创建掩码叠加
        mask_overlay = np.zeros_like(image)
        mask_overlay[mask > 0] = [0, 0, 255]
        image = cv2.addWeighted(image, 0.7, mask_overlay, 0.3, 0)
    
    cv2.imwrite(str(save_dir / name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def evaluate_predictions(pred_masks, gt_masks):
    """评估预测结果"""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    metrics_per_image = []
    
    for pred_mask, gt_mask in zip(pred_masks, gt_masks):
        # 确保是numpy数组
        pred_mask = np.array(pred_mask)
        gt_mask = np.array(gt_mask)
        
        # 确保尺寸一致
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask.astype(np.uint8), (gt_mask.shape[1], gt_mask.shape[0])) > 0
        
        # 计算交集和并集
        intersection = np.logical_and(pred_mask, gt_mask)
        union = np.logical_or(pred_mask, gt_mask)
        
        # 计算TP, FP, FN
        tp = np.sum(intersection)
        fp = np.sum(pred_mask) - tp
        fn = np.sum(gt_mask) - tp
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # 计算单张图像的指标
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        iou = tp / (tp + fp + fn + 1e-6)
        
        metrics_per_image.append({
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'iou_score': float(iou),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn)
        })
    
    # 计算总体指标
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou = total_tp / (total_tp + total_fp + total_fn + 1e-6)
    
    return {
        'overall': {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'iou_score': float(iou),
            'total_tp': int(total_tp),
            'total_fp': int(total_fp),
            'total_fn': int(total_fn)
        },
        'per_image': metrics_per_image
    }

def main(args):
    print("\n开始融合模型实验...")
    print(f"YOLO权重: {args.yolo_weights}")
    print(f"UNET权重: {args.unet_weights}")
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if args.debug:
        debug_dir = save_dir / 'debug'
        debug_dir.mkdir(exist_ok=True)
    
    # 加载YOLO模型
    print("\n加载YOLO模型...")
    yolo_model = YOLO(args.yolo_weights)
    
    # 加载UNET配置和模型
    print("\n加载UNET模型...")
    with open(args.unet_config) as f:
        unet_cfg = yaml.safe_load(f)
    
    device = torch.device(args.device)
    unet_model = create_model(unet_cfg)
    unet_model.load_state_dict(torch.load(args.unet_weights, map_location=device))
    unet_model = unet_model.to(device)
    unet_model.eval()
    
    # 创建融合模型
    print("\n创建融合模型...")
    fusion_model = DecisionFusion(
        yolo_conf_threshold=args.yolo_conf,
        unet_conf_threshold=args.unet_conf,
        iou_threshold=args.iou_thres,
        min_area=args.min_area
    )
    
    # 准备数据转换
    transform = get_val_transforms(unet_cfg)
    
    # 获取测试图像列表
    image_paths = sorted(Path(args.source).glob('*.jpeg'))
    print(f"\n找到{len(image_paths)}张测试图像")
    
    # 获取真实标签掩码
    gt_dir = Path(args.source).parent / 'masks'
    
    # 处理每张图像
    results = []
    pred_masks = []
    gt_masks = []
    
    for image_path in tqdm(image_paths, desc='Processing images'):
        # 读取图像和真实标签
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # 调整图像大小为512x512用于模型预测
        image_resized = cv2.resize(image, (512, 512))
        
        gt_mask = cv2.imread(str(gt_dir / f"{image_path.stem}.png"), cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.resize(gt_mask, (512, 512))
        gt_mask = (gt_mask > 0).astype(np.uint8)
        
        # YOLO预测
        yolo_results = yolo_model(image_resized)[0]
        yolo_boxes = yolo_results.boxes.data.cpu().numpy()
        
        # UNET预测
        transformed = transform(image=image_resized)
        x = transformed['image'].unsqueeze(0).to(device)
        with torch.no_grad():
            unet_pred = torch.sigmoid(unet_model(x))
            unet_mask = unet_pred.squeeze().cpu().numpy()
        
        # 融合预测结果
        fusion_result = fusion_model.fuse_predictions(
            yolo_boxes=yolo_boxes,
            unet_mask=unet_mask,
            image_shape=(512, 512)  # 使用调整后的尺寸
        )
        
        # 将结果调整回原始尺寸
        fusion_result['mask'] = cv2.resize(fusion_result['mask'].astype(np.uint8), (w, h)) > 0
        fusion_result['yolo_mask'] = cv2.resize(fusion_result['yolo_mask'].astype(np.uint8), (w, h)) > 0
        fusion_result['unet_mask'] = cv2.resize(fusion_result['unet_mask'].astype(np.uint8), (w, h)) > 0
        
        # 保存预测掩码和真实标签用于评估
        pred_masks.append(fusion_result['mask'])
        gt_masks.append(gt_mask)
        
        # 可视化结果
        vis_image = fusion_model.visualize_fusion(image, fusion_result)
        
        # 保存结果
        save_name = image_path.stem
        cv2.imwrite(
            str(save_dir / f'{save_name}_fusion.jpg'),
            cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        )
        
        # 保存调试信息
        if args.debug:
            # 保存原始图像
            save_debug_image(debug_dir, f'{save_name}_original.jpg', image)
            
            # 保存真实标签
            save_debug_image(debug_dir, f'{save_name}_gt.jpg', image, gt_mask)
            
            # 保存YOLO预测
            save_debug_image(debug_dir, f'{save_name}_yolo.jpg', image, fusion_result['yolo_mask'])
            
            # 保存UNET预测
            save_debug_image(debug_dir, f'{save_name}_unet.jpg', image, fusion_result['unet_mask'])
            
            # 保存UNET原始预测（概率图）
            cv2.imwrite(
                str(debug_dir / f'{save_name}_unet_prob.jpg'),
                (unet_mask * 255).astype(np.uint8)
            )
        
        # 记录结果
        results.append({
            'image_name': save_name,
            'num_defects': fusion_result['num_defects'],
            'confidences': fusion_result['confidences']
        })
    
    # 评估结果
    print("\n计算评估指标...")
    metrics = evaluate_predictions(pred_masks, gt_masks)
    
    print("\n融合模型评估结果:")
    for metric_name, value in metrics['overall'].items():
        print(f"{metric_name}: {value}")
    
    # 保存评估结果
    metrics['results'] = results
    with open(save_dir / 'fusion_results.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f'\n结果保存到 {save_dir}')
    if args.debug:
        print(f'调试信息保存到 {debug_dir}')

if __name__ == '__main__':
    args = parse_args()
    main(args) 