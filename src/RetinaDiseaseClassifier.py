import torch
import torch.nn as nn
import timm

class RetinaDiseaseClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super(RetinaDiseaseClassifier, self).__init__()
        # Where we define all the parts of the model
        self.base_model = timm.create_model('resnet50', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        out_size = 2048
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_size, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output
    