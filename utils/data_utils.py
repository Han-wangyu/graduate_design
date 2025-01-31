import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def parse_voc_xml(xml_path: str) -> Dict:
    """解析VOC格式的XML标注文件

    Args:
        xml_path: XML文件路径

    Returns:
        包含标注信息的字典
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    annotation = {}
    
    # 获取图像信息
    size = root.find('size')
    annotation['width'] = int(size.find('width').text)
    annotation['height'] = int(size.find('height').text)
    
    # 获取所有目标
    objects = []
    for obj in root.findall('object'):
        obj_dict = {}
        obj_dict['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_dict['bbox'] = [
            float(bbox.find('xmin').text),
            float(bbox.find('ymin').text),
            float(bbox.find('xmax').text),
            float(bbox.find('ymax').text)
        ]
        objects.append(obj_dict)
    
    annotation['objects'] = objects
    return annotation


class PCBAoIDataset(Dataset):
    """PCB-AoI数据集加载器"""
    
    def __init__(self, 
                 data_dir: str,
                 img_size: int = 640,
                 transform = None,
                 is_train: bool = True):
        """初始化数据集

        Args:
            data_dir: 数据集根目录
            img_size: 图像大小
            transform: 数据增强转换
            is_train: 是否为训练集
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.transform = transform
        self.is_train = is_train
        
        # 读取index.txt文件
        index_file = os.path.join(data_dir, 'index.txt')
        self.samples = []
        with open(index_file, 'r') as f:
            for line in f:
                img_path, xml_path = line.strip().split()
                self.samples.append({
                    'img_path': os.path.join(data_dir, img_path.lstrip('./')),
                    'xml_path': os.path.join(data_dir, xml_path.lstrip('./'))
                })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        sample = self.samples[idx]
        
        # 读取图像
        img = cv2.imread(sample['img_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 读取标注
        annotation = parse_voc_xml(sample['xml_path'])
        
        # 准备目标检测格式的标签
        labels = []
        for obj in annotation['objects']:
            bbox = obj['bbox']
            # 转换为YOLO格式 [x_center, y_center, width, height]
            x_center = (bbox[0] + bbox[2]) / 2 / annotation['width']
            y_center = (bbox[1] + bbox[3]) / 2 / annotation['height']
            width = (bbox[2] - bbox[0]) / annotation['width']
            height = (bbox[3] - bbox[1]) / annotation['height']
            labels.append([0, x_center, y_center, width, height])  # 0表示缺陷类别
        
        labels = np.array(labels)
        
        # 应用数据增强
        if self.transform:
            transformed = self.transform(image=img, bboxes=labels[:, 1:], class_labels=labels[:, 0])
            img = transformed['image']
            if len(transformed['bboxes']) > 0:
                labels = np.column_stack([transformed['class_labels'], 
                                       transformed['bboxes']])
            else:
                labels = np.zeros((0, 5))
        
        # 转换为Tensor
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        labels = torch.from_numpy(labels).float()
        
        return img, labels


def create_dataloader(data_dir: str,
                     batch_size: int = 16,
                     img_size: int = 640,
                     transform = None,
                     is_train: bool = True,
                     workers: int = 4):
    """创建数据加载器

    Args:
        data_dir: 数据集根目录
        batch_size: 批次大小
        img_size: 图像大小
        transform: 数据增强转换
        is_train: 是否为训练集
        workers: 数据加载线程数

    Returns:
        数据加载器
    """
    dataset = PCBAoIDataset(
        data_dir=data_dir,
        img_size=img_size,
        transform=transform,
        is_train=is_train
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn
    )


def collate_fn(batch):
    """自定义batch收集函数"""
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    return imgs, labels 