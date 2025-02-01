import argparse
import os
import yaml
import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

from models.unet.model import create_model
from models.fusion import DecisionFusion
from utils.augmentation import get_val_transforms

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', type=str, required=True,
                      help='YOLO模型权重路径')
    parser.add_argument('--unet-weights', type=str, required=True,
                      help='UNET模型权重路径')
    parser.add_argument('--unet-config', type=str, default='configs/unet.yaml',
                      help='UNET配置文件路径')
    parser.add_argument('--source', type=str, required=True,
                      help='测试图像目录或图像路径')
    parser.add_argument('--save-dir', type=str, default='runs/predict_fusion',
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
    return parser.parse_args()

def load_models(args):
    """加载模型"""
    # 加载YOLO模型
    yolo_model = YOLO(args.yolo_weights)
    
    # 加载UNET配置和模型
    with open(args.unet_config) as f:
        unet_cfg = yaml.safe_load(f)
    
    device = torch.device(args.device)
    unet_model = create_model(unet_cfg)
    unet_model.load_state_dict(torch.load(args.unet_weights, map_location=device))
    unet_model = unet_model.to(device)
    unet_model.eval()
    
    return yolo_model, unet_model

def process_image(image_path, yolo_model, unet_model, fusion_model, device, transform):
    """处理单张图像"""
    # 读取图像
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # YOLO预测
    yolo_results = yolo_model(image)[0]
    yolo_boxes = yolo_results.boxes.data.cpu().numpy()
    
    # UNET预测
    transformed = transform(image=image)
    x = torch.from_numpy(transformed['image']).unsqueeze(0).to(device)
    with torch.no_grad():
        unet_pred = torch.sigmoid(unet_model(x))
        unet_mask = unet_pred.squeeze().cpu().numpy()
    
    # 融合预测结果
    fusion_result = fusion_model.fuse_predictions(
        yolo_boxes=yolo_boxes,
        unet_mask=unet_mask,
        image_shape=image.shape[:2]
    )
    
    # 可视化结果
    vis_image = fusion_model.visualize_fusion(image, fusion_result)
    
    return fusion_result, vis_image

def main(args):
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    yolo_model, unet_model = load_models(args)
    
    # 创建融合模型
    fusion_model = DecisionFusion(
        yolo_conf_threshold=args.yolo_conf,
        unet_conf_threshold=args.unet_conf,
        iou_threshold=args.iou_thres,
        min_area=args.min_area
    )
    
    # 准备数据转换
    transform = get_val_transforms({'data': {'img_size': 512}})
    
    # 获取图像列表
    if os.path.isfile(args.source):
        image_paths = [Path(args.source)]
    else:
        image_paths = sorted(Path(args.source).glob('*.jpg')) + \
                     sorted(Path(args.source).glob('*.jpeg')) + \
                     sorted(Path(args.source).glob('*.png'))
    
    # 处理每张图像
    results = []
    for image_path in tqdm(image_paths, desc='Processing images'):
        # 处理图像
        fusion_result, vis_image = process_image(
            image_path, 
            yolo_model, 
            unet_model, 
            fusion_model,
            args.device,
            transform
        )
        
        # 保存结果
        save_name = image_path.stem
        cv2.imwrite(
            str(save_dir / f'{save_name}_fusion.jpg'),
            cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        )
        
        # 记录结果
        results.append({
            'image_name': save_name,
            'num_defects': fusion_result['num_defects'],
            'confidences': fusion_result['confidences']
        })
    
    # 保存汇总结果
    import json
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'Results saved to {save_dir}')

if __name__ == '__main__':
    args = parse_args()
    main(args) 