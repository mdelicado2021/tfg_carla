import cv2
import numpy as np

def extraer_mascara_gt(img_sem_bgr):
    """Extrae la máscara binaria (0 y 1) de líneas de carril desde la paleta CityScapes."""
    img_sem_rgb = cv2.cvtColor(img_sem_bgr, cv2.COLOR_BGR2RGB)
    rango_bajo = np.array([150, 230, 45])
    rango_alto = np.array([160, 240, 55])
    mask_aislada = cv2.inRange(img_sem_rgb, rango_bajo, rango_alto)
    return (mask_aislada > 0).astype(np.uint8)

def calcular_metricas(pred_mask, gt_mask):
    """
    Calcula IoU, F1-Score (Dice), Precision y Recall para máscaras binarias (0 y 1).
    """
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)

    tp = np.logical_and(pred == 1, gt == 1).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()

    eps = 1e-7
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + eps)

    return {
        'iou': float(iou),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall)
    }