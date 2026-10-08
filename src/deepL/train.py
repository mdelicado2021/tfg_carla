import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader import CarlaLaneDataset
from unet_model import UNet
import os

def train():
    # Detectar si hay tarjeta gráfica (CUDA) disponible en tu equipo Linux o usar CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Iniciando entrenamiento en: {device}")

    # 1. Cargar el Dataset
    # Instanciamos la clase que creaste anteriormente
    dataset = CarlaLaneDataset(rgb_dir='dataset/rgb', sem_dir='dataset/semantic')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"Total de lotes (batches) por época: {len(dataloader)}")

    # 2. Inicializar Modelo, Función de Pérdida y Optimizador
    model = UNet(n_channels=3, n_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Learning rate estándar

    # 3. Bucle de Entrenamiento
    num_epochs = 5 # Para la primera prueba usaremos pocas épocas
    
    for epoch in range(num_epochs):
        model.train() # Ponemos el modelo en modo entrenamiento
        epoch_loss = 0

        for batch_idx, (imagenes, mascaras) in enumerate(dataloader):
            # Enviar los tensores a la memoria de la GPU (o CPU)
            imagenes = imagenes.to(device)
            mascaras = mascaras.to(device)

            # Paso 1: Forward (Predicción)
            predicciones = model(imagenes)
            
            # Paso 2: Calcular el error (Loss)
            loss = criterion(predicciones, mascaras)

            # Paso 3: Backward (Retropropagación y ajuste de pesos)
            optimizer.zero_grad() # Limpiar gradientes anteriores
            loss.backward()       # Calcular nuevos gradientes
            optimizer.step()      # Actualizar la red neuronal

            epoch_loss += loss.item()
            
            # Mostrar progreso cada 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] - Batch [{batch_idx+1}/{len(dataloader)}] - Loss: {loss.item():.4f}")

        # Pérdida media de la época completa
        loss_media = epoch_loss / len(dataloader)
        print(f"===> Fin de Epoch [{epoch+1}/{num_epochs}] - Loss Media: {loss_media:.4f}")

    # 4. Exportar los pesos entrenados
    os.makedirs('modelos_guardados', exist_ok=True)
    ruta_guardado = 'modelos_guardados/unet_carla_carriles.pth'
    torch.save(model.state_dict(), ruta_guardado)
    print(f"\nEntrenamiento finalizado. Modelo guardado en: {ruta_guardado}")

if __name__ == '__main__':
    train()