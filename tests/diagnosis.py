import torch
from PIL import Image
import torchvision.transforms as transforms

from sys import path
path.append('../src')
from GlaucomaDiagnoser import GlaucomaDiagnoser

# Define the model path
model_path = '../models/efficientnet_b0/retinai_efficientnet_b0_1.0.0.pth'
mean = [75.44004655992492/255, 47.15671788712379/255, 25.58542044342586/255]
std = [69.76790132458073/255, 45.55243618334717/255, 26.263278919301026/255]

def predict_single_image(model, image_path, device):
    # Define the transformation for the input image
    IMAGE_SIZE = 224
    data_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
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
        print(outputs)
        _, predicted = torch.max(outputs.data, 1)
        print(predicted)
        
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
    single_image_path = '../data/Glaucoma Images/Validation/Glaucoma_Positive/605.jpg' 

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
