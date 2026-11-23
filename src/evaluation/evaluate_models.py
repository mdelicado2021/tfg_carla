import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import csv
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

# Rutas absolutas para importar módulos del repositorio
DIR_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(DIR_RAIZ, 'src', 'deepL'))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unet_model import UNet
from metrics import extraer_mascara_gt, calcular_metricas

DIR_RGB = os.path.join(DIR_RAIZ, 'eval_dataset', 'rgb')
DIR_SEM = os.path.join(DIR_RAIZ, 'eval_dataset', 'semantic')
DIR_PLOTS = os.path.join(DIR_RAIZ, 'plots')

def predecir_thresholding(img_rgb):
    """Método clásico de referencia (umbral de luminosidad en la mitad inferior)."""
    gris = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, mascara = cv2.threshold(gris, 200, 1, cv2.THRESH_BINARY)
    mascara[:int(mascara.shape[0] * 0.5), :] = 0  # ROI inferior
    return mascara

def predecir_unet(modelo, img_rgb, size_wh, device):
    img_small = cv2.resize(img_rgb, size_wh)
    tensor = torch.from_numpy(img_small.transpose((2, 0, 1))).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
        salida = modelo(tensor)
        mask_small = (torch.sigmoid(salida) > 0.5).squeeze().cpu().numpy().astype(np.uint8)

    return cv2.resize(mask_small, (800, 600), interpolation=cv2.INTER_NEAREST)

def main():
    os.makedirs(DIR_PLOTS, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluando modelos en: {device}")

    imagenes = sorted(os.listdir(DIR_RGB))
    if not imagenes:
        print("[!] No hay imágenes en eval_dataset/rgb. Ejecuta primero capture_eval_dataset.py")
        return

    # 1. Cargar U-Net Casera
    unet_casera = UNet(n_channels=3, n_classes=1).to(device)
    unet_casera.load_state_dict(
        torch.load(os.path.join(DIR_RAIZ, 'saved_models', 'unet_carla_carriles.pth'),
                   map_location=device, weights_only=True)
    )
    unet_casera.eval()

    # 2. Cargar U-Net SMP (ResNet-18)
    unet_smp = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1).to(device)
    unet_smp.load_state_dict(
        torch.load(os.path.join(DIR_RAIZ, 'saved_models', 'unet_smp_resnet18.pth'),
                   map_location=device, weights_only=True)
    )
    unet_smp.eval()

    acumulado = {
        'Thresholding': [],
        'U-Net (Casera)': [],
        'U-Net SMP (ResNet-18)': []
    }

    print(f"Procesando {len(imagenes)} fotogramas de validación...")
    for idx, nombre in enumerate(imagenes, 1):
        img_bgr = cv2.imread(os.path.join(DIR_RGB, nombre))
        img_sem_bgr = cv2.imread(os.path.join(DIR_SEM, nombre))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt = extraer_mascara_gt(img_sem_bgr)

        # Evaluar cada modelo
        acumulado['Thresholding'].append(calcular_metricas(predecir_thresholding(img_rgb), gt))
        acumulado['U-Net (Casera)'].append(calcular_metricas(predecir_unet(unet_casera, img_rgb, (400, 300), device), gt))
        acumulado['U-Net SMP (ResNet-18)'].append(calcular_metricas(predecir_unet(unet_smp, img_rgb, (384, 288), device), gt))

        if idx % 30 == 0 or idx == len(imagenes):
            print(f"  Evaluadas {idx}/{len(imagenes)} imágenes")

    # Calcular medias e imprimir tabla
    resumen = {}
    print("\n================ RESULTADOS DE CALIDAD ================")
    print(f"{'Modelo':<25} | {'IoU':>7} | {'F1-Score':>8} | {'Precision':>9} | {'Recall':>7}")
    print("-" * 68)

    ruta_csv = os.path.join(DIR_PLOTS, 'metricas_calidad.csv')
    with open(ruta_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Modelo', 'IoU', 'F1-Score', 'Precision', 'Recall'])

        for nombre_modelo, lista_m in acumulado.items():
            m_iou = np.mean([x['iou'] for x in lista_m])
            m_f1 = np.mean([x['f1'] for x in lista_m])
            m_prec = np.mean([x['precision'] for x in lista_m])
            m_rec = np.mean([x['recall'] for x in lista_m])

            resumen[nombre_modelo] = (m_iou, m_f1, m_prec, m_rec)
            print(f"{nombre_modelo:<25} | {m_iou:7.4f} | {m_f1:8.4f} | {m_prec:9.4f} | {m_rec:7.4f}")
            writer.writerow([nombre_modelo, f"{m_iou:.4f}", f"{m_f1:.4f}", f"{m_prec:.4f}", f"{m_rec:.4f}"])

    # Gráfica comparativa IoU vs F1-Score
    nombres = list(resumen.keys())
    ious = [resumen[n][0] for n in nombres]
    f1s = [resumen[n][1] for n in nombres]

    x = np.arange(len(nombres))
    ancho = 0.35

    plt.figure(figsize=(10, 6))
    b1 = plt.bar(x - ancho/2, ious, ancho, label='IoU', color='#24305e')
    b2 = plt.bar(x + ancho/2, f1s, ancho, label='F1-Score (Dice)', color='#2a9d8f')

    for bars in (b1, b2):
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.015,
                     f'{yval:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.ylim(0, 1.12)
    plt.xticks(x, nombres, fontsize=11)
    plt.title('Comparativa de Calidad de Segmentación de Carriles (CARLA)', fontsize=13, pad=15)
    plt.ylabel('Puntuación (0.0 - 1.0)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    ruta_pdf = os.path.join(DIR_PLOTS, 'comparativa_calidad.pdf')
    plt.savefig(ruta_pdf, format='pdf', bbox_inches='tight')
    plt.show()
    print(f"\nGráfica guardada en: {ruta_pdf}")
    print(f"Tabla CSV guardada en: {ruta_csv}")

if __name__ == '__main__':
    main()