import torch
import torch.nn as nn
import timm
import numpy as np

class GlaucomaDiagnoser(nn.Module):
    def __init__(self, num_classes=1, base_model='efficientnet_b0', dropout_rate=0.2, 
                 freeze_full_blocks=0, freeze_segment_blocks=0, segment_bias=0.7):
        """
        Args:
            segment_bias (float): Value between 0 and 1 indicating initial bias towards 
                                segmented image (default: 0.7)
        """
        super(GlaucomaDiagnoser, self).__init__()
        
        assert 0 <= segment_bias <= 1, "segment_bias must be between 0 and 1"
        self.segment_bias = segment_bias
        
        # Create two parallel base models
        self.full_model = timm.create_model(base_model, pretrained=True)
        self.segment_model = timm.create_model(base_model, pretrained=True)
        
        # Freeze blocks for full image path
        if freeze_full_blocks > 0:
            total_blocks = len(self.full_model.blocks)
            blocks_to_freeze = min(freeze_full_blocks, total_blocks)
            print(f"Freezing {blocks_to_freeze} out of {total_blocks} blocks in full image path")
            
            for idx, block in enumerate(self.full_model.blocks):
                if idx < blocks_to_freeze:
                    for param in block.parameters():
                        param.requires_grad = False
        
        # Freeze blocks for segmented image path
        if freeze_segment_blocks > 0:
            total_blocks = len(self.segment_model.blocks)
            blocks_to_freeze = min(freeze_segment_blocks, total_blocks)
            print(f"Freezing {blocks_to_freeze} out of {total_blocks} blocks in segment image path")
            
            for idx, block in enumerate(self.segment_model.blocks):
                if idx < blocks_to_freeze:
                    for param in block.parameters():
                        param.requires_grad = False
        
        # Remove the original classifiers
        self.out_channels = self.full_model.num_features
        self.full_model.classifier = nn.Identity()
        self.segment_model.classifier = nn.Identity()
        
        # Global pooling
        self.full_model.global_pool = nn.AdaptiveAvgPool2d(1)
        self.segment_model.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Quality assessment modules
        self.quality_assessor = nn.Sequential(
            nn.Linear(self.out_channels, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Enhanced attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.out_channels * 2, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 2)
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.out_channels * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Initialize attention bias
        with torch.no_grad():
            # Set initial bias in the last layer of attention mechanism
            last_layer = self.attention[-1]
            # Adjust biases to favor segmented image according to segment_bias
            last_layer.bias.data = torch.tensor(
                [np.log(1 - segment_bias), np.log(segment_bias)], 
                dtype=torch.float32
            )

    def get_trainable_params(self):
        """Returns the number of trainable parameters in each component."""
        full_params = sum(p.numel() for p in self.full_model.parameters() if p.requires_grad)
        segment_params = sum(p.numel() for p in self.segment_model.parameters() if p.requires_grad)
        classifier_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        attention_params = sum(p.numel() for p in self.attention.parameters() if p.requires_grad)
        quality_params = sum(p.numel() for p in self.quality_assessor.parameters() if p.requires_grad)
        
        return {
            'full_model': full_params,
            'segment_model': segment_params,
            'classifier': classifier_params,
            'attention': attention_params,
            'quality_assessor': quality_params,
            'total': full_params + segment_params + classifier_params + attention_params + quality_params
        }
    
    def compute_attention_weights(self, full_features, segment_features):
        """
        Compute attention weights with quality assessment and bias.
        Returns both raw attention weights and quality scores for monitoring.
        """
        # Assess quality of segmented image
        segment_quality = self.quality_assessor(segment_features)
        
        # Compute attention scores
        combined_features = torch.cat((full_features, segment_features), dim=1)
        attention_logits = self.attention(combined_features)
        
        # Apply quality-aware attention
        # If segment quality is low, shift attention to full image
        attention_weights = torch.softmax(attention_logits, dim=1)
        quality_adjusted_weights = attention_weights.clone()
        quality_adjusted_weights[:, 1] *= segment_quality.squeeze()  # Adjust segment weight by quality
        quality_adjusted_weights[:, 0] += (1 - segment_quality.squeeze()) * attention_weights[:, 1]  # Redistribute to full image
        
        # Renormalize weights
        quality_adjusted_weights = quality_adjusted_weights / quality_adjusted_weights.sum(dim=1, keepdim=True)
        
        return quality_adjusted_weights, segment_quality
        
    def forward(self, full_img, segment_img):
        # Process both images through their respective paths
        full_features = self.full_model(full_img)
        segment_features = self.segment_model(segment_img)
        
        # Flatten features
        full_features = full_features.view(full_features.size(0), -1)
        segment_features = segment_features.view(segment_features.size(0), -1)
        
        # Calculate attention weights with quality assessment
        attention_weights, quality_scores = self.compute_attention_weights(full_features, segment_features)
        
        # Apply attention weights
        weighted_features = torch.cat((
            full_features * attention_weights[:, 0].unsqueeze(1),
            segment_features * attention_weights[:, 1].unsqueeze(1)
        ), dim=1)
        
        # Final classification
        output = self.classifier(weighted_features)
        
        if self.training:
            # During training, return quality scores for monitoring
            return output, quality_scores
        else:
            # During inference, just return the prediction
            return output
