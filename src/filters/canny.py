import carla
import random
import pygame
import numpy as np
import cv2
import time
import os

historial_tiempos = []

def procesar_camara_frontal(imagen, superficie_pygame):
    t_inicio = time.perf_counter()

    # 1. Extraer la imagen original bruta
    array = np.frombuffer(imagen.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (imagen.height, imagen.width, 4))
    img_bgr = array[:, :, :3]
    img_rgb_original = img_bgr[:, :, ::-1] 
    
    alto, ancho = img_bgr.shape[:2] # 600x800
    
    # ================= PASO 1: REGIÓN DE INTERÉS (ROI) =================
    # Mitad inferior de la cámara
    vertices_roi = np.array([[
        (0, alto),         # Abajo izquierda
        (0, 300),        # Arriba izquierda
        (800, 300),        # Arriba derecha
        (800, alto)        # Abajo derecha
    ]], dtype=np.int32)
    
    mascara_roi = np.zeros((alto, ancho), dtype=np.uint8)
    cv2.fillPoly(mascara_roi, vertices_roi, 255)
    
    # Imagen original recortada
    roi_original_rgb = cv2.bitwise_and(img_rgb_original, img_rgb_original, mask=mascara_roi)
    
    # ================= PASO 2: THRESHOLDING (Filtro de Color) =================
    img_hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    blanco_bajo = np.array([0, 200, 0])
    blanco_alto = np.array([255, 255, 255])
    
    mascara_blanca = cv2.inRange(img_hls, blanco_bajo, blanco_alto)
    mascara_filtrada_roi = cv2.bitwise_and(mascara_blanca, mascara_roi)
    
    imagen_thresholding = np.dstack((mascara_filtrada_roi, mascara_filtrada_roi, mascara_filtrada_roi))
    
    # ================= PASO 3: CANNY (Detección de Bordes) =================
    # Aplicamos Canny sobre la máscara binaria para extraer solo las siluetas finas
    bordes = cv2.Canny(mascara_filtrada_roi, 50, 150)

    t_fin = time.perf_counter()
    tiempo_ms = (t_fin - t_inicio) * 1000.0
    historial_tiempos.append(tiempo_ms)
    print(f"Latencia Percepción (Canny): {tiempo_ms:.2f} ms")
    
    imagen_canny = np.dstack((bordes, bordes, bordes))
    
    # ================= RENDERIZAR EN CUADRÍCULA 2x2 =================
    # Arriba Izquierda: Original
    superficie_pygame.blit(pygame.surfarray.make_surface(img_rgb_original.swapaxes(0, 1)), (0, 0))
    
    # Arriba Derecha: ROI a color
    superficie_pygame.blit(pygame.surfarray.make_surface(roi_original_rgb.swapaxes(0, 1)), (800, 0))
    
    # Abajo Izquierda: Thresholding + ROI (Manchas sólidas)
    superficie_pygame.blit(pygame.surfarray.make_surface(imagen_thresholding.swapaxes(0, 1)), (0, 600))
    
    # Abajo Derecha: Canny (Esqueletos finos)
    superficie_pygame.blit(pygame.surfarray.make_surface(imagen_canny.swapaxes(0, 1)), (800, 600))

def main():
    pygame.init()
    # Ventana 2x2 (1600x1200)
    pantalla = pygame.display.set_mode((1600, 1200))
    pygame.display.set_caption("CARLA - Pipeline 2x2: Original | ROI | Threshold | Canny")

    vehiculo = None
    camara_frontal = None
    original_settings = None

    try:
        print("Conectando con CARLA...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        map_name = 'Town04'
        world = client.get_world()

        # Solo recarga el mundo si no estamos ya en Town06
        if not world.get_map().name.endswith(map_name):
            print(f"Cargando el mapa {map_name}. El simulador se congelará unos segundos...")
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
        bp_camara.set_attribute('image_size_x', '800')
        bp_camara.set_attribute('image_size_y', '600')
        
        transform_frontal = carla.Transform(carla.Location(x=2.0, z=1.5), carla.Rotation(pitch=-15))
        camara_frontal = world.spawn_actor(bp_camara, transform_frontal, attach_to=vehiculo)
        camara_frontal.listen(lambda image: procesar_camara_frontal(image, pantalla))
        
        print("Rodando. Pulsa Ctrl+C en la terminal para salir.")
        
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
        path_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'times'))
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
            
        path_file = os.path.join(path_dir, 'canny_times.txt')
        np.savetxt(path_file, historial_tiempos)

        if world is not None and original_settings is not None:
            world.apply_settings(original_settings)
        if camara_frontal is not None: camara_frontal.destroy()
        if vehiculo is not None: vehiculo.destroy()
        pygame.quit()

if __name__ == '__main__':
    main()