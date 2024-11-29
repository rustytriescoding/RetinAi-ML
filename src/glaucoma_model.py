import torch
import torch.nn as nn
import timm

class GlaucomaDiagnoser(nn.Module):
    def __init__(self, base_model='resnet50', dropout_rate=0, freeze_blocks=0):
        super(GlaucomaDiagnoser, self).__init__()
        
        self.model = timm.create_model(base_model, pretrained=True, in_chans=1)

        # if hasattr(self.model, 'conv1'):
        #     original_conv = self.model.conv1.weight.data
        #     self.model.conv1.weight.data = original_conv.sum(dim=1, keepdim=True) / 3.0
        # elif hasattr(self.model, 'conv_stem'):
        #     original_conv = self.model.conv_stem.weight.data
        #     self.model.conv_stem.weight.data = original_conv.sum(dim=1, keepdim=True) / 3.0
        
        # Unfreeze all parameters initially
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Handle different model architectures
        if 'resnet' in base_model:
            self._setup_resnet(freeze_blocks)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
            
        else:  # Default handling for other models (efficientnet etc)
            self._setup_default(freeze_blocks)
            num_features = self.model.num_features
            self.model.classifier = nn.Identity()

        # Medical imaging-optimized classifier architecture
        # self.classifier = nn.Sequential(
        #     nn.Linear(num_features, 512),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(512),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(256),
        #     nn.Dropout(dropout_rate/2),
        #     nn.Linear(256, 1)
        # )
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 4),
            nn.ReLU(),
            nn.BatchNorm1d(4),
            nn.Dropout(dropout_rate/2),
            nn.Linear(4, 1)
        )

    def _setup_resnet(self, freeze_blocks):
        """Configure ResNet specific freezing"""
        if freeze_blocks > 0:
            layers_to_freeze = ['layer1', 'layer2', 'layer3', 'layer4'][:freeze_blocks]
            print(f"Freezing ResNet layers: {layers_to_freeze}")
            
            for name, param in self.model.named_parameters():
                if any(layer in name for layer in layers_to_freeze):
                    param.requires_grad = False


    def _setup_default(self, freeze_blocks):
        """Configure default model freezing for other architectures"""
        if freeze_blocks > 0:
            if hasattr(self.model, 'blocks'):
                total_blocks = len(self.model.blocks)
                blocks_to_freeze = min(freeze_blocks, total_blocks)
                print(f"Freezing {blocks_to_freeze} out of {total_blocks} blocks")
                
                for idx, block in enumerate(self.model.blocks):
                    if idx < blocks_to_freeze:
                        for param in block.parameters():
                            param.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        output = self.classifier(x)
        return output
