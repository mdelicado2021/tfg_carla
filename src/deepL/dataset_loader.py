import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class CarlaLaneDataset(Dataset):
    def __init__(self, rgb_dir, sem_dir):
        self.rgb_dir = rgb_dir
        self.sem_dir = sem_dir
        # Listamos y ordenamos los archivos para que la 000001.png RGB coincida con la 000001.png Semántica
        self.images = sorted(os.listdir(rgb_dir))
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        # ================= 1. PROCESAR IMAGEN RGB (Entrada de la Red) =================
        rgb_path = os.path.join(self.rgb_dir, img_name)
        img_rgb = cv2.imread(rgb_path)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        
        # PyTorch espera los canales al principio (Canales, Alto, Ancho) y valores normalizados entre 0 y 1
        img_tensor = torch.from_numpy(img_rgb.transpose((2, 0, 1))).float() / 255.0
        
        # ================= 2. PROCESAR MÁSCARA SEMÁNTICA (Ground Truth) =================
        sem_path = os.path.join(self.sem_dir, img_name)
        img_sem = cv2.imread(sem_path)
        img_sem_rgb = cv2.cvtColor(img_sem, cv2.COLOR_BGR2RGB)
        
        # La paleta CityScapes de CARLA pinta las líneas de carril de color RGB exacto (157, 234, 50)
        # Usamos un rango muy estrecho para atrapar solo ese color verde claro
        rango_bajo = np.array([150, 230, 45])
        rango_alto = np.array([160, 240, 55])
        mask_aislada = cv2.inRange(img_sem_rgb, rango_bajo, rango_alto)
        
        # Convertimos la máscara a binaria: 0.0 (Fondo) y 1.0 (Línea)
        mask_binaria = (mask_aislada > 0).astype(np.float32)
        
        # Añadimos la dimensión del canal para PyTorch: (1, Alto, Ancho)
        mask_tensor = torch.from_numpy(mask_binaria).unsqueeze(0) 
        
        return img_tensor, mask_tensor

def main():
    # Inicializamos nuestro Dataset apuntando a las carpetas que creaste
    dataset = CarlaLaneDataset(rgb_dir='dataset/rgb', sem_dir='dataset/semantic')
    
    print(f"Total de imágenes en el dataset: {len(dataset)}")
    
    # Creamos el DataLoader (empaqueta los datos en lotes y los mezcla)
    # batch_size=4 significa que la red neuronal procesará 4 imágenes a la vez
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Extraemos el primer lote de prueba
    imagenes_batch, mascaras_batch = next(iter(dataloader))
    
    print(f"Forma del tensor RGB: {imagenes_batch.shape} -> (Batch, Canales, Alto, Ancho)")
    print(f"Forma del tensor Máscara: {mascaras_batch.shape} -> (Batch, Canales, Alto, Ancho)")
    
    # ================= VISUALIZAR EL RESULTADO =================
    # Extraemos la primera imagen del lote para verla en matplotlib
    img_vista = imagenes_batch[0].numpy().transpose((1, 2, 0)) # Devolvemos a (Alto, Ancho, Canales) para dibujar
    mask_vista = mascaras_batch[0].numpy().squeeze() # Quitamos la dimensión del canal (Alto, Ancho)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img_vista)
    ax[0].set_title("Tensor RGB (Entrada)")
    ax[0].axis('off')
    
    ax[1].imshow(mask_vista, cmap='gray')
    ax[1].set_title("Tensor Binario (0 y 1)")
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()