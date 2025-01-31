import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def convert_box(size, box):
    """将VOC格式的边界框转换为YOLO格式

    Args:
        size: 图像尺寸 (width, height)
        box: VOC格式的边界框 [xmin, ymin, xmax, ymax]

    Returns:
        YOLO格式的边界框 [x_center, y_center, width, height]
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]


def convert_voc_to_yolo(data_dir, output_dir):
    """将VOC格式的数据集转换为YOLO格式

    Args:
        data_dir: VOC格式数据集目录
        output_dir: 输出目录
    """
    # 创建输出目录
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # 读取index.txt
    index_file = os.path.join(data_dir, 'index.txt')
    with open(index_file, 'r') as f:
        lines = f.readlines()

    # 处理每个样本
    for line in tqdm(lines, desc=f'Converting {Path(data_dir).name}'):
        img_path, xml_path = line.strip().split()
        img_path = os.path.join(data_dir, img_path.lstrip('./'))
        xml_path = os.path.join(data_dir, xml_path.lstrip('./'))

        # 解析XML文件
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取图像尺寸
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        # 复制图像文件
        img_name = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(images_dir, img_name))

        # 转换标注
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        txt_path = os.path.join(labels_dir, txt_name)

        with open(txt_path, 'w') as f:
            for obj in root.findall('object'):
                # 我们只有一个类别：缺陷
                cls_id = 0
                
                xmlbox = obj.find('bndbox')
                box = [
                    float(xmlbox.find('xmin').text),
                    float(xmlbox.find('ymin').text),
                    float(xmlbox.find('xmax').text),
                    float(xmlbox.find('ymax').text)
                ]
                
                # 转换为YOLO格式
                bb = convert_box((width, height), box)
                
                # 写入文件
                f.write(f"{cls_id} {' '.join([str(a) for a in bb])}\n")


def main():
    # 转换训练集
    convert_voc_to_yolo(
        'data/dataset/train_data',
        'data/dataset/train_data'
    )
    
    # 转换测试集
    convert_voc_to_yolo(
        'data/dataset/test_data',
        'data/dataset/test_data'
    )
    
    print('数据集转换完成！')


if __name__ == '__main__':
    main() 