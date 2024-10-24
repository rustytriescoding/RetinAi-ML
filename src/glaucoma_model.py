import torch
import torch.nn as nn
import timm

class GlaucomaDiagnoser(nn.Module):
    def __init__(self, num_classes=2, base_model='efficientnet_b0', dropout_rate=0.2, freeze_blocks=0):
        super(GlaucomaDiagnoser, self).__init__()
        self.base_model = timm.create_model(base_model, pretrained=True)
        
        # Freeze blocks
        if freeze_blocks > 0:
            total_blocks = len(self.base_model.blocks)
            blocks_to_freeze = min(freeze_blocks, total_blocks)
            print(f"Freezing {blocks_to_freeze} out of {total_blocks} blocks")
            
            for idx, block in enumerate(self.base_model.blocks):
                if idx < blocks_to_freeze:
                    for param in block.parameters():
                        param.requires_grad = False
                    print(f"Block {idx} frozen")

        # Get the output size based on the base model
        self.out_channels = self.base_model.num_features
        
        # Remove the original classifier
        self.base_model.classifier = nn.Identity()
        self.base_model.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier layer
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.out_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate/2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Get features from base model
        features = self.base_model(x)
        # Flatten the features
        features = features.view(features.size(0), -1)
        # Pass through classifier
        return self.classifier(features)

