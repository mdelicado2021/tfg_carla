import carla
import random
import pygame
import numpy as np
import cv2
import time
import os
import sys
import shutil
from huggingface_hub import hf_hub_download

DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
DIRECTORIO_RAIZ = os.path.abspath(os.path.join(DIRECTORIO_ACTUAL, '..', '..'))
RUTA_UFLD_SRC = os.path.join(DIRECTORIO_RAIZ, 'src', 'external', 'autonomous-driving-carla', 'src')

sys.path.append(RUTA_UFLD_SRC)

try:
    from ultrafastLaneDetector import UltrafastLaneDetector, ModelType
except ImportError as e:
    print(f"[!] Error importando UFLD. Revisa la estructura del repositorio: {e}")
    sys.exit(1)

historial_tiempos = []

def inicializar_modelo_ufld():
    ruta_modelos = os.path.join(DIRECTORIO_RAIZ, 'saved_models')
    if not os.path.exists(ruta_modelos):
        os.makedirs(ruta_modelos)
        
    archivo_pesos = os.path.join(ruta_modelos, 'ufld_carla_best.pth')
    
    if not os.path.exists(archivo_pesos):
        print("Descargando pesos SOTA desde Hugging Face...")
        archivo_descargado = hf_hub_download(repo_id="jkdxbns/autonomous-driving-carla", filename="ufld_carla_best.pth")
        
        # Extracción del archivo físico desde la caché de Hugging Face
        shutil.copyfile(archivo_descargado, archivo_pesos)
    else:
        print("Pesos UFLD encontrados en saved_models.")

    print("Cargando modelo neuronal UFLD en RAM (CPU)...")
    
    modelo = UltrafastLaneDetector(
        archivo_pesos, 
        ModelType.TUSIMPLE, 
        use_gpu=False
    )
    return modelo

def procesar_camara_ufld(imagen, superficie_pygame, modelo):
    array = np.frombuffer(imagen.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (imagen.height, imagen.width, 4))
    img_bgr = array[:, :, :3]
    
    t_inicio = time.perf_counter()
    
    lane_visualization = modelo.detect_lanes(img_bgr)
    lanes_points = modelo.lanes_points
    
    t_fin = time.perf_counter()
    tiempo_ms = (t_fin - t_inicio) * 1000.0
    historial_tiempos.append(tiempo_ms)
    print(f"Latencia Percepción (UFLD GPU): {tiempo_ms:.2f} ms")
    
    img_rgb_lienzo = lane_visualization[:, :, ::-1]
    
    superficie_pygame.blit(pygame.surfarray.make_surface(img_rgb_lienzo.swapaxes(0, 1)), (0, 0))

def main():
    modelo_ufld = inicializar_modelo_ufld()

    pygame.init()
    pantalla = pygame.display.set_mode((800, 288)) 
    pygame.display.set_caption("CARLA - Benchmark Inferencia UFLD (SOTA)")

    # Inicializacion de variables de entorno para evitar UnboundLocalError en el bloque finally
    vehiculo = None
    camara_frontal = None
    original_settings = None
    world = None

    try:
        print("Conectando con CARLA...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        map_name = 'Town04'
        world = client.get_world()

        if not world.get_map().name.endswith(map_name):
            world = client.load_world(map_name)
        
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True 
        settings.fixed_delta_seconds = 0.05 
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        
        blueprint_library = world.get_blueprint_library()
        modelo_coche = blueprint_library.filter('model3')[0]
        puntos_spawn = world.get_map().get_spawn_points()
        
        vehiculo = world.spawn_actor(modelo_coche, random.choice(puntos_spawn))
        vehiculo.set_autopilot(True, traffic_manager.get_port())
        
        bp_camara = blueprint_library.find('sensor.camera.rgb')
        bp_camara.set_attribute('image_size_x', '1640')
        bp_camara.set_attribute('image_size_y', '590')
        bp_camara.set_attribute('fov', '150')
        
        transform_frontal = carla.Transform(carla.Location(x=1.5, z=2.4))
        camara_frontal = world.spawn_actor(bp_camara, transform_frontal, attach_to=vehiculo)
        
        camara_frontal.listen(lambda image: procesar_camara_ufld(image, pantalla, modelo_ufld))
        
        print("Rodando UFLD. Recabando 100 fotogramas...")
        running = True
        while running:
            world.tick()
            
            if len(historial_tiempos) >= 100:
                print("Límite de 100 fotogramas alcanzado. Guardando resultados...")
                running = False
                
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            pygame.display.flip()

    except KeyboardInterrupt:
        pass
    finally:
        print("Guardando registro de tiempos...")
        path_dir = os.path.join(DIRECTORIO_RAIZ, 'times')
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
            
        path_file = os.path.join(path_dir, 'ufld_times.txt')
        np.savetxt(path_file, historial_tiempos)

        if world is not None and original_settings is not None:
            world.apply_settings(original_settings)
        if camara_frontal is not None: camara_frontal.destroy()
        if vehiculo is not None: vehiculo.destroy()
        pygame.quit()

if __name__ == '__main__':
    main()