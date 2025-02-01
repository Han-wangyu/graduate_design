import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import argparse
import yaml
from models.unet.model import create_model
from utils.metrics import compute_metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='experiments/unet/run_20250201_214043/weights/best.pt',
                      help='UNET模型权重路径')
    parser.add_argument('--config', type=str, default='configs/experiment_unet.yaml',
                      help='UNET模型配置文件')
    parser.add_argument('--source', type=str, required=True,
                      help='输入图像路径或目录')
    parser.add_argument('--save-dir', type=str, default='results',
                      help='保存结果的目录')
    parser.add_argument('--device', default='',
                      help='cuda设备，例如：0或0,1,2,3或cpu')
    parser.add_argument('--conf', type=float, default=0.5,
                      help='置信度阈值')
    parser.add_argument('--min-area', type=int, default=200,
                      help='最小面积阈值')
    parser.add_argument('--debug', action='store_true',
                      help='保存调试信息')
    return parser.parse_args()

def load_model(weights_path, config_path, device):
    # 加载配置文件
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 创建模型
    model = create_model(config)
    
    # 加载权重
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model

def process_image(img_path, model, device, conf_thres, min_area, save_dir, debug=False):
    """处理单张图像
    
    Args:
        img_path: 图像路径
        model: 加载的模型
        device: 设备
        conf_thres: 置信度阈值
        min_area: 最小面积阈值
        save_dir: 保存目录
        debug: 是否保存调试信息
    
    Returns:
        predictions: 预测结果列表，每个预测包含：
            - bbox: [x1, y1, x2, y2] 格式的边界框
            - confidence: 置信度
            - class: 类别 (0)
    """
    # 加载图像
    img0 = cv2.imread(str(img_path))
    if img0 is None:
        print(f'无法读取图像: {img_path}')
        return None
    
    # 预处理图像
    img = cv2.resize(img0, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std
    
    # 转换为张量
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).float().unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        pred = model(img)
        pred = torch.sigmoid(pred)
        pred = pred.squeeze().cpu().numpy()
    
    # 将预测掩码缩放到原始图像大小
    pred_resized = cv2.resize(pred, (img0.shape[1], img0.shape[0]))
    
    # 改进的自适应阈值处理
    binary = np.zeros_like(pred_resized, dtype=np.uint8)
    
    # 1. 计算全局统计信息
    global_mean = np.mean(pred_resized)
    global_std = np.std(pred_resized)
    
    # 2. 找到高置信度区域
    high_conf_mask = pred_resized > conf_thres
    if np.any(high_conf_mask):
        # 3. 计算高置信度区域的统计信息
        high_conf_mean = np.mean(pred_resized[high_conf_mask])
        high_conf_std = np.std(pred_resized[high_conf_mask])
        
        # 4. 使用OTSU算法在高置信度区域找到最优阈值
        high_conf_values = (pred_resized[high_conf_mask] * 255).astype(np.uint8)
        if len(high_conf_values) > 1:
            otsu_thresh, _ = cv2.threshold(high_conf_values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu_thresh = otsu_thresh / 255.0
        else:
            otsu_thresh = conf_thres
        
        # 5. 自适应阈值计算
        # 根据局部区域的统计特性调整阈值
        local_thresh = min(
            max(
                conf_thres,
                otsu_thresh,
                global_mean + global_std,
                high_conf_mean - high_conf_std
            ),
            0.95  # 上限防止过度筛选
        )
        
        # 6. 应用阈值，但保持高置信度区域
        binary[pred_resized > local_thresh] = 1
        binary[pred_resized > max(0.8, conf_thres + 0.2)] = 1  # 保留非常高置信度的区域
    else:
        # 如果没有高置信度区域，使用基于全局统计的阈值
        local_thresh = min(max(conf_thres, global_mean + global_std * 1.5), 0.95)
        binary[pred_resized > local_thresh] = 1
    
    # 改进的形态学操作
    # 使用两个不同大小的核进行处理
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # 先用小核进行闭运算去除小孔
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small)
    # 再用大核进行开运算去除噪点
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_large)
    
    # 保存结果
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 生成可视化结果
    heatmap = (pred_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img0, 0.7, heatmap, 0.3, 0)
    
    # 寻找并过滤缺陷区域
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = img0.copy()
    
    defects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            # 计算最小外接矩形
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=np.int32)
            
            # 计算区域内的平均置信度
            mask = np.zeros_like(pred_resized)
            cv2.drawContours(mask, [cnt], 0, 1, -1)
            conf = float(np.mean(pred_resized[mask > 0]))
            
            # 计算轮廓的复杂度（周长/面积比）
            perimeter = cv2.arcLength(cnt, True)
            complexity = perimeter * perimeter / (4 * np.pi * area)
            
            # 过滤条件：
            # 1. 置信度要足够高
            # 2. 形状不能太不规则（复杂度不能太高）
            # 3. 长宽比不能太极端
            # 4. 添加面积比例限制
            total_area = img0.shape[0] * img0.shape[1]
            area_ratio = area / total_area
            
            # 计算轮廓的密实度（实际面积与最小外接矩形面积之比）
            rect_area = rect[1][0] * rect[1][1]
            solidity = area / (rect_area + 1e-6)
            
            # 计算轮廓的圆形度
            circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
            
            if (conf >= conf_thres and 
                complexity < 2.8 and  # 收紧形状复杂度阈值
                0.15 < rect[1][0] / (rect[1][1] + 1e-6) < 8.0 and  # 收紧长宽比限制
                0.0002 < area_ratio < 0.05 and  # 收紧面积比例限制
                0.4 < solidity < 0.95 and  # 调整密实度范围
                circularity > 0.2):  # 添加圆形度限制
                
                # 绘制缺陷区域
                cv2.drawContours(result, [box], 0, (0, 0, 255), 2)
                
                # 获取标准格式的边界框坐标
                x, y, w, h = cv2.boundingRect(cnt)
                defects.append({
                    'bbox': [x, y, x + w, y + h],
                    'confidence': conf,
                    'class': 0
                })
    
    # 保存结果
    base_name = Path(img_path).stem
    cv2.imwrite(str(save_path / f'{base_name}_result.jpg'), result)
    cv2.imwrite(str(save_path / f'{base_name}_heatmap.jpg'), heatmap)
    cv2.imwrite(str(save_path / f'{base_name}_overlay.jpg'), overlay)
    
    if debug:
        cv2.imwrite(str(save_path / f'{base_name}_binary.jpg'), binary * 255)
        cv2.imwrite(str(save_path / f'{base_name}_pred.jpg'), (pred_resized * 255).astype(np.uint8))
    
    return defects

def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if args.debug:
        debug_dir = save_dir / 'debug'
        debug_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    model = load_model(args.weights, args.config, device)
    print(f'模型已加载: {args.weights}')
    
    # 处理输入路径
    source = Path(args.source)
    if source.is_file():
        image_files = [source]
    else:
        image_files = sorted(source.glob('*.jpg')) + sorted(source.glob('*.jpeg')) + sorted(source.glob('*.png'))
    
    print(f'找到{len(image_files)}张图像')
    
    # 准备评估结果
    all_predictions = []
    all_ground_truths = []
    
    # 处理每张图像
    for img_file in tqdm(image_files, desc='处理图像'):
        # 加载图像以获取尺寸
        img0 = cv2.imread(str(img_file))
        if img0 is None:
            print(f'无法读取图像: {img_file}')
            continue
            
        predictions = process_image(
            img_file,
            model,
            device,
            args.conf,
            args.min_area,
            args.save_dir,
            args.debug
        )
        
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
        
        all_predictions.append(predictions if predictions else [])
        all_ground_truths.append(ground_truths)
        
        if predictions:
            print(f'\n图像 {img_file.name} 检测到 {len(predictions)} 个缺陷:')
            for i, d in enumerate(predictions, 1):
                x1, y1, x2, y2 = d['bbox']
                print(f'缺陷 {i}:')
                print(f'  位置: [{x1}, {y1}, {x2}, {y2}]')
                print(f'  置信度: {d["confidence"]:.4f}')
    
    # 计算评估指标
    print('\n计算评估指标...')
    metrics = compute_metrics(all_predictions, all_ground_truths)
    
    print('\nUNET模型评估结果:')
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