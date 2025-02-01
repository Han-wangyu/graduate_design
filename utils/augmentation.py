import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(cfg):
    """获取训练数据增强"""
    return A.Compose([
        A.Resize(512, 512),  # 调整到512x512（32的倍数）
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_val_transforms(cfg):
    """获取验证数据增强"""
    return A.Compose([
        A.Resize(512, 512),  # 调整到512x512（32的倍数）
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ]) 