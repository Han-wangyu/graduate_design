import argparse
import os
import yaml
import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from models.unet.model import create_model
from utils.augmentation import get_val_transforms
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/unet.yaml',
                      help='配置文件路径')
    parser.add_argument('--weights', type=str, default='runs/train/exp/weights/best.pt',
                      help='模型权重路径')
    parser.add_argument('--source', type=str, default='data/dataset/test_data/images',
                      help='测试图像目录')
    parser.add_argument('--save-dir', type=str, default='runs/predict',
                      help='结果保存目录')
    parser.add_argument('--device', type=str, default='cpu',
                      help='预测设备')
    return parser.parse_args()

def calculate_metrics(pred_masks, true_masks, threshold=0.5):
    """计算评估指标"""
    pred_binary = (pred_masks > threshold).astype(np.uint8)
    true_binary = (true_masks > threshold).astype(np.uint8)
    
    # 计算IoU
    intersection = np.logical_and(pred_binary, true_binary).sum()
    union = np.logical_or(pred_binary, true_binary).sum()
    iou = intersection / (union + 1e-6)
    
    # 计算F1分数
    f1 = f1_score(true_binary.flatten(), pred_binary.flatten())
    
    # 计算准确率
    acc = accuracy_score(true_binary.flatten(), pred_binary.flatten())
    
    return {
        'iou': iou,
        'f1': f1,
        'accuracy': acc
    }

def main(args):
    # 加载配置
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 创建模型
    model = create_model(cfg).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取验证转换
    transform = get_val_transforms(cfg)
    
    # 获取所有测试图像
    img_files = [f for f in os.listdir(args.source) if f.endswith(('.jpg', '.jpeg', '.png'))]
    mask_dir = os.path.join(os.path.dirname(args.source), 'masks')
    
    # 存储所有预测结果用于计算指标
    all_preds = []
    all_masks = []
    
    print('开始预测...')
    for img_file in tqdm(img_files):
        # 读取图像
        img_path = os.path.join(args.source, img_file)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取对应的真实掩码
        mask_path = os.path.join(mask_dir, img_file.replace('.jpeg', '.png'))
        if os.path.exists(mask_path):
            true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0
        
        # 保存原始图像尺寸
        orig_h, orig_w = image.shape[:2]
        
        # 应用转换
        transformed = transform(image=image)
        image = transformed['image']
        
        # 预测
        with torch.no_grad():
            x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
            pred = model(x)
            pred = torch.sigmoid(pred)
            pred = pred.squeeze().cpu().numpy()
        
        # 调整预测掩码大小以匹配原始图像
        pred = cv2.resize(pred, (orig_w, orig_h))
        
        if os.path.exists(mask_path):
            all_preds.append(pred)
            all_masks.append(true_mask)
        
        # 可视化结果
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
        plt.title('原始图像')
        plt.axis('off')
        
        plt.subplot(132)
        if os.path.exists(mask_path):
            plt.imshow(true_mask, cmap='gray')
            plt.title('真实掩码')
        else:
            plt.title('无真实掩码')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(pred, cmap='gray')
        plt.title('预测掩码')
        plt.axis('off')
        
        plt.savefig(save_dir / f'{img_file.split(".")[0]}_result.png')
        plt.close()
    
    # 计算整体指标
    if all_preds and all_masks:
        metrics = calculate_metrics(
            np.array(all_preds),
            np.array(all_masks),
            threshold=cfg['val']['threshold']
        )
        
        print('\n评估指标:')
        print(f'IoU: {metrics["iou"]:.4f}')
        print(f'F1 Score: {metrics["f1"]:.4f}')
        print(f'Accuracy: {metrics["accuracy"]:.4f}')
        
        # 保存指标
        with open(save_dir / 'metrics.txt', 'w') as f:
            for k, v in metrics.items():
                f.write(f'{k}: {v:.4f}\n')

if __name__ == '__main__':
    args = parse_args()
    main(args) 