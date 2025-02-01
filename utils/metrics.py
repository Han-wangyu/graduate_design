import numpy as np

def compute_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def compute_metrics(all_predictions, all_ground_truths, iou_threshold=0.5):
    """计算评估指标"""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou = 0
    total_matches = 0

    for predictions, ground_truths in zip(all_predictions, all_ground_truths):
        # 初始化匹配标记
        matched_gt = [False] * len(ground_truths)
        
        # 对每个预测框
        for pred in predictions:
            pred_box = pred['bbox']
            max_iou = 0
            max_idx = -1
            
            # 找到最佳匹配的真实框
            for i, gt in enumerate(ground_truths):
                if matched_gt[i]:
                    continue
                    
                gt_box = gt['bbox']
                iou = compute_iou(pred_box, gt_box)
                
                if iou > max_iou:
                    max_iou = iou
                    max_idx = i
            
            # 如果找到匹配且IoU超过阈值
            if max_idx >= 0 and max_iou >= iou_threshold:
                total_tp += 1
                matched_gt[max_idx] = True
                total_iou += max_iou
                total_matches += 1
            else:
                total_fp += 1
        
        # 计算未匹配的真实框（漏检）
        total_fn += sum(1 for x in matched_gt if not x)
    
    # 计算指标
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    iou_score = total_iou / total_matches if total_matches > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'iou_score': iou_score,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn
    } 