import torch
import torch.nn as nn
import timm
import numpy as np

class GlaucomaDiagnoser(nn.Module):
    def __init__(self, base_model='efficientnet_b0', dropout_rate=0,
                 freeze_full_blocks=0, freeze_segment_blocks=0):

        super(GlaucomaDiagnoser, self).__init__()
        
        self.full_model = timm.create_model(base_model, pretrained=True)
        self.segment_model = timm.create_model(base_model, pretrained=True)
        
        # Freeze blocks for full image
        if freeze_full_blocks > 0:
            total_blocks = len(self.full_model.blocks)
            blocks_to_freeze = min(freeze_full_blocks, total_blocks)
            print(f"Freezing {blocks_to_freeze} out of {total_blocks} blocks in full image path")
            
            for idx, block in enumerate(self.full_model.blocks):
                if idx < blocks_to_freeze:
                    for param in block.parameters():
                        param.requires_grad = False
        
        # Freeze blocks for segmented image
        if freeze_segment_blocks > 0:
            total_blocks = len(self.segment_model.blocks)
            blocks_to_freeze = min(freeze_segment_blocks, total_blocks)
            print(f"Freezing {blocks_to_freeze} out of {total_blocks} blocks in segment image path")
            
            for idx, block in enumerate(self.segment_model.blocks):
                if idx < blocks_to_freeze:
                    for param in block.parameters():
                        param.requires_grad = False
       
        self.num_features = self.full_model.num_features
       
        self.full_model.classifier = nn.Identity()
        self.segment_model.classifier = nn.Identity()   
       
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features * 2, 32),  
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 4),
            nn.ReLU(),
            nn.Linear(4, 1)  
        )

    def forward(self, full_img, segment_img):
        full_features = self.full_model(full_img)
        segment_features = self.segment_model(segment_img)
        
        combined = torch.cat((full_features, segment_features), dim=1)
        
        return self.classifier(combined)
