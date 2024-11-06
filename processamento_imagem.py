from PIL import Image
import torch
import torchvision.transforms as transforms
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize((32, 32)),  
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
])

def predict_single_image(image_tensor, model):
    model.eval()  
    with torch.no_grad():  
        image_tensor = image_tensor.to(device)  
        output = model(image_tensor)  
        _, predicted = torch.max(output, 1)  
        return predicted.item()  

# Caminho da imagem para a qual queremos fazer a previsão
image_path = 'C:/Users/Karol/Documents/Python projeto/Bases/ICIAR2018_BACH_Challenge/ICIAR2018_BACH_Challenge/Photos/BACH_Preparado/200x200/Invasive/iv001.tif'

image = Image.open(image_path)
image_tensor = transform(image).unsqueeze(0)  


model_path = 'C:/Users/Karol/Documents/GitHub/python_analysis'  
file_name = 'modelo.pth'  

cnn = CNN() 
cnn.load_state_dict(torch.load(os.path.join(model_path, file_name)))  
cnn = cnn.to(device) 

# Prever a classe da imagem
predicted_class = predict_single_image(image_tensor, cnn)

classes = ['Benign', 'InSitu', 'Invasive', 'Normal']  

# Imprimir a classe prevista
print('Classe prevista:', classes[predicted_class])
