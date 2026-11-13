import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Navegación absoluta: partimos de src/ y subimos un nivel a la raíz
DIRECTORIO_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

CARPETA_TIEMPOS = os.path.join(DIRECTORIO_RAIZ, 'times')
CARPETA_GRAFICAS = os.path.join(DIRECTORIO_RAIZ, 'plots')

if not os.path.exists(CARPETA_GRAFICAS):
    os.makedirs(CARPETA_GRAFICAS)

# Diccionario con los nombres exactos de los archivos
algoritmos = {
    'Thresholding': 'thresholding_times.txt',
    'Canny': 'canny_times.txt',
    'Hough': 'hough_times.txt',
    'Sliding Windows': 'sliding_times.txt',
    'U-Net (CPU)': 'unet_times.txt',
    'UFLD (CPU)': 'ufld_times.txt'
}

medias_ms = []
nombres_validos = []

# ================= GRÁFICAS INDIVIDUALES =================
for nombre, nombre_archivo in algoritmos.items():
    ruta_archivo = os.path.join(CARPETA_TIEMPOS, nombre_archivo)
    
    if os.path.exists(ruta_archivo):
        datos = np.loadtxt(ruta_archivo)
        
        # Omitir primer frame para evitar picos de inicialización
        if len(datos) > 1:
            datos = datos[1:] 
            
        media = np.mean(datos)
        medias_ms.append(media)
        nombres_validos.append(nombre)
        
        plt.figure(figsize=(8, 4))
        plt.plot(datos, color='#374785', alpha=0.8, linewidth=1.5, label='Latencia por fotograma')
        plt.axhline(media, color='red', linestyle='--', linewidth=2, label=f'Media: {media:.2f} ms')
        
        plt.title(f'Estabilidad Temporal de Latencia: {nombre}', fontsize=12)
        plt.xlabel('Fotograma', fontsize=10)
        plt.ylabel('Tiempo (ms)', fontsize=10)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        nombre_pdf = f'latencia_{nombre.replace(" ", "_").lower()}.pdf'
        ruta_pdf = os.path.join(CARPETA_GRAFICAS, nombre_pdf)
        plt.savefig(ruta_pdf, format='pdf', bbox_inches='tight')
        plt.close()
        print(f" Generada: {ruta_pdf}")
    else:
        print(f"[!] No se encontró la ruta: {ruta_archivo}")

# ================= GRÁFICA COMPARATIVA (MEDIAS) =================
if nombres_validos:
    plt.figure(figsize=(10, 6))
    
    colores_base = ['#a8d0e6', '#f8e9a1', '#f76c6c', '#374785', '#24305e', '#16697a']
    colores = colores_base[:len(nombres_validos)]
    
    bars = plt.bar(nombres_validos, medias_ms, color=colores)

    for bar in bars:
        yval = bar.get_height()
        desplazamiento = max(medias_ms) * 0.02
        plt.text(bar.get_x() + bar.get_width()/2, yval + desplazamiento, 
                 f'{yval:.2f} ms', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.title('Comparativa de Latencia Media en Pipelines de Percepción 3D', fontsize=14, pad=15)
    plt.ylabel('Tiempo de ejecución medio (ms)', fontsize=12)
    plt.xlabel('Algoritmo de Percepción', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    ruta_pdf_comparativa = os.path.join(CARPETA_GRAFICAS, 'comparativa_latencias_media.pdf')
    plt.savefig(ruta_pdf_comparativa, format='pdf', bbox_inches='tight')
    plt.show()
    print(f"\n Generada gráfica comparativa global: {ruta_pdf_comparativa}")