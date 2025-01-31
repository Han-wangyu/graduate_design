import os
import cv2
import numpy as np
from xml.etree import ElementTree as ET
from tqdm import tqdm

def create_mask_from_xml(xml_path, img_shape):
    """从VOC XML文件创建掩码"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 创建空白掩码
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    
    # 遍历所有目标
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        
        # 在掩码上绘制矩形
        cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), 255, -1)
    
    return mask

def convert_dataset(base_path):
    """转换整个数据集"""
    # 创建掩码目录
    for split in ['train_data', 'test_data']:
        img_dir = os.path.join(base_path, split, 'images')
        xml_dir = os.path.join(base_path, split, 'Annotations')
        mask_dir = os.path.join(base_path, split, 'masks')
        os.makedirs(mask_dir, exist_ok=True)
        
        # 获取所有图像文件
        img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpeg')]
        
        print(f'Converting {split} set...')
        for img_file in tqdm(img_files):
            # 读取图像获取尺寸
            img_path = os.path.join(img_dir, img_file)
            img = cv2.imread(img_path)
            
            # 获取对应的XML文件
            xml_file = img_file.replace('.jpeg', '.xml')
            xml_path = os.path.join(xml_dir, xml_file)
            
            if os.path.exists(xml_path):
                # 创建掩码
                mask = create_mask_from_xml(xml_path, img.shape)
                
                # 保存掩码
                mask_file = img_file.replace('.jpeg', '.png')
                mask_path = os.path.join(mask_dir, mask_file)
                cv2.imwrite(mask_path, mask)

if __name__ == '__main__':
    dataset_path = 'data/dataset'
    convert_dataset(dataset_path)
    print('Conversion completed!') 