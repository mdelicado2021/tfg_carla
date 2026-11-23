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
    'Sliding\nWindows': 'sliding_times.txt',
    'U-Net\n(CPU)': 'unet_cpu_times.txt',
    'U-Net\n(GPU)': 'unet_gpu_times.txt',
    'U-Net SMP\n(CPU)': 'unet_smp_cpu_times.txt',
    'U-Net SMP\n(GPU)': 'unet_smp_gpu_times.txt',
    'UFLD\n(CPU)': 'ufld_cpu_times.txt',
    'UFLD\n(GPU)': 'ufld_gpu_times.txt',
}

medias_dict = {}

# ================= 1. GRÁFICAS INDIVIDUALES =================
for nombre, nombre_archivo in algoritmos.items():
    ruta_archivo = os.path.join(CARPETA_TIEMPOS, nombre_archivo)
    
    if os.path.exists(ruta_archivo):
        datos = np.loadtxt(ruta_archivo)
        
        # Omitir primer frame para evitar picos de inicialización
        if len(datos) > 1:
            datos = datos[1:] 
            
        media = np.mean(datos)
        medias_dict[nombre] = media
        
        nombre_limpio = nombre.replace("\n", " ")
        
        plt.figure(figsize=(8, 4))
        plt.plot(datos, color='#374785', alpha=0.8, linewidth=1.5, label='Latencia por fotograma')
        plt.axhline(media, color='red', linestyle='--', linewidth=2, label=f'Media: {media:.2f} ms')
        
        plt.title(f'Estabilidad Temporal de Latencia: {nombre_limpio}', fontsize=12)
        plt.xlabel('Fotograma', fontsize=10)
        plt.ylabel('Tiempo (ms)', fontsize=10)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        nombre_pdf = f'latencia_{nombre_limpio.replace(" ", "_").replace("(", "").replace(")", "").lower()}.pdf'
        ruta_pdf = os.path.join(CARPETA_GRAFICAS, nombre_pdf)
        plt.savefig(ruta_pdf, format='pdf', bbox_inches='tight')
        plt.close()
        print(f" Generada: {ruta_pdf}")
    else:
        print(f"[!] No se encontró la ruta: {ruta_archivo}")


# ================= 2. GRÁFICA COMPARATIVA GLOBAL =================
if medias_dict:
    nombres_validos = list(medias_dict.keys())
    medias_ms = list(medias_dict.values())

    plt.figure(figsize=(14, 6))
    colores_base = [
        '#a8d0e6', '#f8e9a1', '#f76c6c', '#374785',
        '#24305e', '#16697a', '#8d99ae', '#2a9d8f', '#e9c46a', '#e76f51'
    ]
    colores = colores_base[:len(nombres_validos)]
    
    bars = plt.bar(nombres_validos, medias_ms, color=colores, width=0.6)

    for bar in bars:
        yval = bar.get_height()
        desplazamiento = max(medias_ms) * 0.02
        plt.text(bar.get_x() + bar.get_width()/2, yval + desplazamiento, 
                 f'{yval:.2f} ms', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.ylim(0, max(medias_ms) * 1.15)
    plt.title('Comparativa Global de Latencia Media en Pipelines de Percepción 3D', fontsize=14, pad=15)
    plt.ylabel('Tiempo de ejecución medio (ms)', fontsize=12)
    plt.xlabel('Algoritmo de Percepción', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    ruta_pdf_global = os.path.join(CARPETA_GRAFICAS, 'comparativa_latencias_media.pdf')
    plt.savefig(ruta_pdf_global, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"\n Generada gráfica comparativa global: {ruta_pdf_global}")


# ================= 3. GRÁFICA SOLO MÉTODOS CLÁSICOS =================
claves_clasicos = ['Thresholding', 'Canny', 'Hough', 'Sliding\nWindows']
clasicos_nombres = [k for k in claves_clasicos if k in medias_dict]
clasicos_medias = [medias_dict[k] for k in clasicos_nombres]

if clasicos_nombres:
    plt.figure(figsize=(9, 5.5))
    colores_clasicos = ['#a8d0e6', '#f8e9a1', '#f76c6c', '#374785'][:len(clasicos_nombres)]
    
    bars = plt.bar(clasicos_nombres, clasicos_medias, color=colores_clasicos, width=0.55)

    for bar in bars:
        yval = bar.get_height()
        desplazamiento = max(clasicos_medias) * 0.025
        plt.text(bar.get_x() + bar.get_width()/2, yval + desplazamiento, 
                 f'{yval:.2f} ms', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.ylim(0, max(clasicos_medias) * 1.18)
    plt.title('Comparativa de Latencia Media: Métodos Clásicos de Visión Artificial', fontsize=13, pad=15)
    plt.ylabel('Tiempo de ejecución medio (ms)', fontsize=12)
    plt.xlabel('Algoritmo Clásico', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    ruta_pdf_clasicos = os.path.join(CARPETA_GRAFICAS, 'comparativa_clasicos.pdf')
    plt.savefig(ruta_pdf_clasicos, format='pdf', bbox_inches='tight')
    plt.show()
    print(f" Generada gráfica métodos clásicos: {ruta_pdf_clasicos}")


# ================= 4. GRÁFICA DEEP LEARNING: CPU vs GPU =================
pares_dl = [
    ('U-Net (Casera)', 'U-Net\n(CPU)', 'U-Net\n(GPU)'),
    ('U-Net SMP\n(ResNet-18)', 'U-Net SMP\n(CPU)', 'U-Net SMP\n(GPU)'),
    ('UFLD', 'UFLD\n(CPU)', 'UFLD\n(GPU)'),
]

dl_etiquetas = []
dl_cpu = []
dl_gpu = []

for etiqueta, clave_cpu, clave_gpu in pares_dl:
    if clave_cpu in medias_dict and clave_gpu in medias_dict:
        dl_etiquetas.append(etiqueta)
        dl_cpu.append(medias_dict[clave_cpu])
        dl_gpu.append(medias_dict[clave_gpu])

if dl_etiquetas:
    x = np.arange(len(dl_etiquetas))
    ancho = 0.35

    plt.figure(figsize=(10, 6))
    bars_cpu = plt.bar(x - ancho/2, dl_cpu, ancho, label='CPU', color='#24305e')
    bars_gpu = plt.bar(x + ancho/2, dl_gpu, ancho, label='GPU (RTX 3050)', color='#2a9d8f')

    max_dl = max(max(dl_cpu), max(dl_gpu))
    desplazamiento = max_dl * 0.02

    for bar_c, bar_g in zip(bars_cpu, bars_gpu):
        val_c = bar_c.get_height()
        val_g = bar_g.get_height()
        speedup = val_c / val_g if val_g > 0 else 0

        plt.text(bar_c.get_x() + bar_c.get_width()/2, val_c + desplazamiento,
                 f'{val_c:.2f} ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.text(bar_g.get_x() + bar_g.get_width()/2, val_g + desplazamiento,
                 f'{val_g:.2f} ms\n({speedup:.1f}x)', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1d7065')

    plt.ylim(0, max_dl * 1.18)
    plt.xticks(x, dl_etiquetas, fontsize=11)
    plt.title('Comparativa de Latencia Media en Deep Learning: CPU vs. GPU', fontsize=13, pad=15)
    plt.ylabel('Tiempo de ejecución medio (ms)', fontsize=12)
    plt.xlabel('Arquitectura Neuronal', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    ruta_pdf_dl = os.path.join(CARPETA_GRAFICAS, 'comparativa_dl_cpu_vs_gpu.pdf')
    plt.savefig(ruta_pdf_dl, format='pdf', bbox_inches='tight')
    plt.show()
    print(f" Generada gráfica Deep Learning (CPU vs GPU): {ruta_pdf_dl}")