import torch
import torchvision.transforms as transforms
import os
from PIL import Image
from torch import nn
import torch.nn.functional as F

# Definição da rede neural (CNN)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)  # Aumentado o número de filtros para 32
        self.pool = nn.MaxPool2d(2, 2)    # Max Pooling com kernel 2x2
        self.conv2 = nn.Conv2d(32, 64, 5) # Aumentado o número de filtros para 64
        self.fc1 = nn.Linear(64 * 5 * 5, 256)  # Ajustado o tamanho da entrada para 64 * 5 * 5 e aumentado para 256 neurônios
        self.dropout = nn.Dropout(p=0.5)  # Dropout para regularização
        self.fc2 = nn.Linear(256, 128)  # Segunda camada totalmente conectada com 128 neurônios
        self.fc3 = nn.Linear(128, 4)    # 4 classes de saída: Benign, InSitu, Invasive, Normal

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))  # Usando Leaky ReLU ao invés de ReLU
        x = self.pool(F.leaky_relu(self.conv2(x)))  # Segunda camada convolucional com Leaky ReLU
        x = torch.flatten(x, 1)  # Achatar as dimensões para passar para a camada linear
        x = F.leaky_relu(self.fc1(x))  # Primeira camada totalmente conectada com Leaky ReLU
        x = self.dropout(x)            # Aplicar Dropout
        x = F.leaky_relu(self.fc2(x))  # Segunda camada totalmente conectada com Leaky ReLU
        x = self.fc3(x)  # Camada de saída (sem ativação aqui porque será usada no critério de perda)
        return x

# Definindo o dispositivo (GPU se disponível, senão CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Função de predição
def predict_single_image(image_tensor, model):
    model.eval()  
    with torch.no_grad():  
        image_tensor = image_tensor.to(device)  
        output = model(image_tensor)  
        _, predicted = torch.max(output, 1)  
        return predicted.item()  

# Função para processar a imagem e fazer a previsão
def process_image():
    # Caminho da imagem para previsão
    image_path = 'C:/cancer/python_analysis/imagem/imagem2.jpeg'
    
    # Carregar e transformar a imagem
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    ])
    image_tensor = transform(image).unsqueeze(0)
    
    # Caminho do modelo
    model_path = 'C:/cancer/python_analysis'  
    file_name = 'modelo.pth'  
    
    # Inicializar e carregar o modelo
    cnn = CNN()  
    cnn.load_state_dict(torch.load(os.path.join(model_path, file_name)))  
    cnn = cnn.to(device) 
    
    # Prever a classe da imagem
    predicted_class = predict_single_image(image_tensor, cnn)
    
    # Classes possíveis
    classes = ['Benign', 'InSitu', 'Invasive', 'Normal']
    predicted_label = classes[predicted_class]
    
    # Retornar a mensagem formatada
    return f'Imagem processada com sucesso. Classe prevista: {predicted_label}'
