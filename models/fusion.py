import numpy as np
import torch
import cv2
from typing import Dict, Tuple, List

class DecisionFusion:
    """决策融合模块，用于组合YOLO和UNET的预测结果"""
    
    def __init__(self, 
                 yolo_conf_threshold: float = 0.25,
                 unet_conf_threshold: float = 0.5,
                 iou_threshold: float = 0.5,
                 min_area: int = 100):
        """
        初始化决策融合模块
        
        Args:
            yolo_conf_threshold: YOLO检测置信度阈值
            unet_conf_threshold: UNET分割置信度阈值
            iou_threshold: 两个模型预测区域的IoU阈值
            min_area: 最小缺陷区域面积
        """
        self.yolo_conf_threshold = yolo_conf_threshold
        self.unet_conf_threshold = unet_conf_threshold
        self.iou_threshold = iou_threshold
        self.min_area = min_area
    
    def _convert_yolo_to_mask(self, 
                             yolo_boxes: np.ndarray, 
                             image_shape: Tuple[int, int]) -> Tuple[np.ndarray, List[float]]:
        """将YOLO的边界框转换为掩码

        Args:
            yolo_boxes: YOLO预测的边界框 [x1, y1, x2, y2, conf]
            image_shape: 图像尺寸 (H, W)

        Returns:
            与图像同尺寸的二值掩码和对应的置信度列表
        """
        mask = np.zeros(image_shape, dtype=np.uint8)
        confidences = []
        
        for box in yolo_boxes:
            if box[4] >= self.yolo_conf_threshold:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)
                confidences.append(float(box[4]))
        
        return mask, confidences
    
    def _process_unet_mask(self, 
                          unet_mask: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """处理UNET的预测掩码

        Args:
            unet_mask: UNET预测的概率掩码

        Returns:
            二值化的掩码和对应区域的置信度列表
        """
        # 二值化
        binary_mask = (unet_mask > self.unet_conf_threshold).astype(np.uint8)
        confidences = []
        
        # 处理连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        # 创建新的掩码
        processed_mask = np.zeros_like(binary_mask)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_area:
                region_mask = (labels == i)
                region_conf = float(np.mean(unet_mask[region_mask]))
                if region_conf >= self.unet_conf_threshold:
                    processed_mask[region_mask] = 1
                    confidences.append(region_conf)
        
        return processed_mask, confidences
    
    def _calculate_region_confidence(self,
                                   yolo_mask: np.ndarray,
                                   yolo_confidences: List[float],
                                   unet_mask: np.ndarray,
                                   unet_confidences: List[float]) -> Tuple[np.ndarray, List[float]]:
        """计算每个区域的置信度

        Args:
            yolo_mask: YOLO预测的掩码
            yolo_confidences: YOLO预测的置信度列表
            unet_mask: UNET预测的掩码
            unet_confidences: UNET预测的置信度列表

        Returns:
            融合后的掩码和每个区域的置信度
        """
        # 初始化融合掩码
        fusion_mask = np.zeros_like(yolo_mask)
        confidences = []
        
        # 合并两个掩码
        combined_mask = (yolo_mask | unet_mask).astype(np.uint8)
        
        # 找到所有连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            combined_mask, 
            connectivity=8
        )
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < self.min_area:
                continue
                
            region_mask = (labels == i).astype(np.uint8)
            
            # 计算与YOLO和UNET预测的重叠
            yolo_overlap = np.sum(region_mask & yolo_mask) / np.sum(region_mask)
            unet_overlap = np.sum(region_mask & unet_mask) / np.sum(region_mask)
            
            # 获取对应区域的最高置信度
            yolo_conf = max(yolo_confidences) if yolo_confidences and yolo_overlap > 0.3 else 0
            unet_conf = max(unet_confidences) if unet_confidences and unet_overlap > 0.3 else 0
            
            # 使用加权平均计算最终置信度
            if yolo_overlap > 0.3 or unet_overlap > 0.3:
                confidence = (yolo_conf * yolo_overlap + unet_conf * unet_overlap) / (yolo_overlap + unet_overlap + 1e-6)
                
                # 如果置信度足够高，保留该区域
                if confidence > min(self.yolo_conf_threshold, self.unet_conf_threshold):
                    fusion_mask |= region_mask
                    confidences.append(confidence)
        
        return fusion_mask, confidences
    
    def fuse_predictions(self,
                        yolo_boxes: np.ndarray,
                        unet_mask: np.ndarray,
                        image_shape: Tuple[int, int]) -> Dict:
        """融合YOLO和UNET的预测结果

        Args:
            yolo_boxes: YOLO预测的边界框 [x1, y1, x2, y2, conf]
            unet_mask: UNET预测的概率掩码
            image_shape: 图像尺寸 (H, W)

        Returns:
            包含融合结果的字典：
            {
                'mask': 融合后的掩码,
                'confidences': 每个区域的置信度,
                'num_defects': 检测到的缺陷数量,
                'yolo_mask': YOLO预测掩码,
                'unet_mask': UNET预测掩码
            }
        """
        # 转换YOLO结果为掩码
        yolo_mask, yolo_confidences = self._convert_yolo_to_mask(yolo_boxes, image_shape)
        
        # 处理UNET掩码
        unet_binary_mask, unet_confidences = self._process_unet_mask(unet_mask)
        
        # 融合预测结果
        fusion_mask, confidences = self._calculate_region_confidence(
            yolo_mask, yolo_confidences,
            unet_binary_mask, unet_confidences
        )
        
        # 统计缺陷数量
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            fusion_mask.astype(np.uint8), 
            connectivity=8
        )
        num_defects = num_labels - 1  # 减去背景
        
        return {
            'mask': fusion_mask,
            'confidences': confidences,
            'num_defects': num_defects,
            'yolo_mask': yolo_mask,
            'unet_mask': unet_binary_mask
        }
    
    def visualize_fusion(self,
                        image: np.ndarray,
                        fusion_result: Dict,
                        save_path: str = None) -> np.ndarray:
        """可视化融合结果

        Args:
            image: 原始图像
            fusion_result: fuse_predictions的返回结果
            save_path: 保存路径（可选）

        Returns:
            可视化结果图像
        """
        # 创建可视化图像
        vis_image = image.copy()
        
        # 创建三个通道的掩码叠加
        mask_overlay = np.zeros_like(image)
        mask_overlay[fusion_result['yolo_mask'] > 0] = [255, 0, 0]   # 蓝色表示YOLO检测
        mask_overlay[fusion_result['unet_mask'] > 0] = [0, 255, 0]   # 绿色表示UNET检测
        mask_overlay[fusion_result['mask'] > 0] = [0, 0, 255]        # 红色表示融合结果
        
        # 叠加到原图
        vis_image = cv2.addWeighted(vis_image, 0.7, mask_overlay, 0.3, 0)
        
        # 添加文本信息
        text = [
            f"Defects: {fusion_result['num_defects']}",
            f"Conf: {np.mean(fusion_result['confidences']):.3f}" if fusion_result['confidences'] else "Conf: 0.000"
        ]
        
        for i, t in enumerate(text):
            cv2.putText(vis_image, t, (10, 30 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 保存结果
        if save_path:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image 