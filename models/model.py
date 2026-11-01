import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from configs.config import  NUM_FRAMES
class HybridVideoClassifier(nn.Module):
    def __init__(self, num_frames=NUM_FRAMES, hidden_dim=512, num_classes=1, freeze_backbone=True, fine_tune_layer4=True):
        super().__init__()

        # -----------------------------
        # 3D ResNet backbone (spatio-temporal)
        # -----------------------------
        self.backbone = r3d_18(pretrained=True)
        self.backbone.fc = nn.Identity()  # remove classifier

        # -----------------------------
        # Freeze backbone except optional layer4 fine-tuning
        # -----------------------------
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False
            if fine_tune_layer4:
                # Unfreeze last residual block
                for param in self.backbone.layer4.parameters():
                    param.requires_grad = True

        # -----------------------------
        # Transformer for temporal modeling
        # -----------------------------
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # -----------------------------
        # Classification head
        # -----------------------------
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x: [B, F, C, H, W] 
        returns logits: [B]
        """
        B, F, C, H, W = x.shape

        # Permute for 3D CNN: B, C, F, H, W
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        # 3D ResNet expects B, C, T, H, W
        features = self.backbone(x)  # outputs [B, 512]

        # Repeat backbone features across frames for temporal transformer
        features = features.unsqueeze(1).repeat(1, F, 1)  # [B, F, 512]

        # Temporal modeling
        features = self.transformer(features)             # [B, F, 512]
        features = features.mean(dim=1)                  # mean over frames -> [B, 512]

        # Classification
        logits = self.classifier(features).squeeze(1)   # [B]
        return logits