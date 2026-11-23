import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

# Asegurar que encuentra dataset_loader.py en el mismo directorio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset_loader import CarlaLaneDataset

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Iniciando entrenamiento U-Net SMP (ResNet-18) en: {device}")

    # 1. Cargar el Dataset
    dataset = CarlaLaneDataset(rgb_dir='dataset/rgb', sem_dir='dataset/semantic')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"Total de lotes (batches) por época: {len(dataloader)}")

    # 2. Inicializar Modelo SMP con Encoder ResNet-18 preentrenado
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    # Con Transfer Learning conviene un learning rate algo más suave que desde cero
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # 3. Bucle de Entrenamiento
    num_epochs = 10  # En GPU tardará muy poco
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (imagenes, mascaras) in enumerate(dataloader):
            imagenes = imagenes.to(device)
            mascaras = mascaras.to(device)

            # Ajustamos a (Alto=288, Ancho=384) -> múltiplos de 32 para ResNet-18
            imagenes = F.interpolate(imagenes, size=(288, 384), mode='bilinear', align_corners=False)
            mascaras = F.interpolate(mascaras, size=(288, 384), mode='nearest')

            # Paso 1: Forward
            predicciones = model(imagenes)
            
            # Paso 2: Loss
            loss = criterion(predicciones, mascaras)

            # Paso 3: Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] - Batch [{batch_idx+1}/{len(dataloader)}] - Loss: {loss.item():.4f}")

        loss_media = epoch_loss / len(dataloader)
        print(f"===> Fin de Epoch [{epoch+1}/{num_epochs}] - Loss Media: {loss_media:.4f}")

    # 4. Exportar los pesos entrenados en saved_models/
    os.makedirs('saved_models', exist_ok=True)
    ruta_guardado = 'saved_models/unet_smp_resnet18.pth'
    torch.save(model.state_dict(), ruta_guardado)
    print(f"\nEntrenamiento finalizado. Modelo guardado en: {ruta_guardado}")

if __name__ == '__main__':
    train()