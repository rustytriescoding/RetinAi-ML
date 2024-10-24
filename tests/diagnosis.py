import torch
from PIL import Image
import torchvision.transforms as transforms

from sys import path
path.append('../src')
from glaucoma_model import GlaucomaDiagnoser

# Define the model path
model_path = '../models/efficientnet_b0/retinai_efficientnet_b0_0.0.1.pth'
mean = [0.21161936893269484, 0.0762942479204578, 0.02436706896535593]
std = [0.1694396363130937, 0.07732758785821338, 0.02954764478795169]

class GammaAdjust:
    def __init__(self, gamma=0.4):
        self.gamma = gamma

    def __call__(self, img):
        return transforms.functional.adjust_gamma(img, self.gamma)

def predict_single_image(model, image_path, device):
    # Define the transformation for the input image
    IMAGE_SIZE = 224
    data_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        GammaAdjust(gamma=0.4),
        transforms.Normalize(mean, std)
    ])
    
    # Load the image
    image = Image.open(image_path)
    image = data_transform(image)
    image = image.unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        
    return predicted.item()  # Return the predicted class

def main():
    # Load the model
    model = 'efficientnet_b0'
    glaucoma_model = GlaucomaDiagnoser(num_classes=2, base_model=model)
    glaucoma_model.load_state_dict(torch.load(model_path, weights_only=False))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    glaucoma_model.to(device)

    # Path to the single image you want to predict
    # single_image_path = '../data/Glaucoma Images/Validation/Glaucoma_Negative/487.jpg' 
    single_image_path = '../data/Glaucoma Images/Validation/Glaucoma_Positive/604.jpg' 

    # Make prediction
    prediction = predict_single_image(glaucoma_model, single_image_path, device)
    
    # Interpret the prediction
    if prediction == 0:
        print("Diagnosis: Normal")
    elif prediction == 1:
        print("Diagnosis: Glaucoma")
    else:
        print("Diagnosis: Unknown")

if __name__ == '__main__':
    main()
