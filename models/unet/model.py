import segmentation_models_pytorch as smp
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, cfg):
        super(UNet, self).__init__()
        self.model = smp.Unet(
            encoder_name=cfg['model']['encoder_name'],
            encoder_weights=cfg['model']['encoder_weights'],
            in_channels=cfg['model']['in_channels'],
            classes=cfg['model']['classes'],
            activation=cfg['model']['activation']
        )
    
    def forward(self, x):
        return self.model(x)

def create_model(cfg):
    """创建UNET模型实例"""
    return UNet(cfg) 