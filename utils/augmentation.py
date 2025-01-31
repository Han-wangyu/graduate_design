import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int = 640):
    """获取训练阶段的数据增强

    Args:
        img_size: 图像大小

    Returns:
        数据增强转换
    """
    return A.Compose([
        A.RandomResizedCrop(
            height=img_size,
            width=img_size,
            scale=(0.8, 1.0),
            ratio=(0.8, 1.2),
            p=0.5
        ),
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.GaussianBlur(p=0.5),
            A.MotionBlur(p=0.5),
        ], p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ], bbox_params=A.BboxParams(
        format='yolo',
        min_visibility=0.3,
        label_fields=['class_labels']
    ))


def get_val_transforms(img_size: int = 640):
    """获取验证阶段的数据增强

    Args:
        img_size: 图像大小

    Returns:
        数据增强转换
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ], bbox_params=A.BboxParams(
        format='yolo',
        min_visibility=0.3,
        label_fields=['class_labels']
    )) 