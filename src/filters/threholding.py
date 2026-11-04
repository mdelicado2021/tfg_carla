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
    
    alto, ancho = img_bgr.shape[:2] # 600x800
    
    # ================= 2. DEFINIR LA REGIÓN DE INTERÉS (ROI) AMPLIADA =================
    vertices_roi = np.array([[
        (0, alto),         
        (0, 300),        
        (800, 300),        
        (800, alto)        
    ]], dtype=np.int32)
    
    mascara_roi = np.zeros((alto, ancho), dtype=np.uint8)
    cv2.fillPoly(mascara_roi, vertices_roi, 255)
    
    # ================= 3. APLICAR THRESHOLDING (Filtro Blanco) =================
    img_hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    blanco_bajo = np.array([0, 200, 0])
    blanco_alto = np.array([255, 255, 255])
    
    mascara_blanca = cv2.inRange(img_hls, blanco_bajo, blanco_alto)
    mascara_filtrada_roi = cv2.bitwise_and(mascara_blanca, mascara_roi)

    t_fin = time.perf_counter()
    tiempo_ms = (t_fin - t_inicio) * 1000.0
    historial_tiempos.append(tiempo_ms)
    print(f"Latencia Percepción (Thresholding Clásico): {tiempo_ms:.2f} ms")
    
    # ================= OPERACIONES ESTÉTICAS (FUERA DEL CRONÓMETRO) =================
    img_rgb_original = img_bgr[:, :, ::-1] 
    roi_original_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=mascara_roi)
    roi_original_rgb = roi_original_bgr[:, :, ::-1]
    imagen_filtrada_rgb = np.dstack((mascara_filtrada_roi, mascara_filtrada_roi, mascara_filtrada_roi)) 
    
    # ================= 4. RENDERIZAR EN 3 PANTALLAS =================
    surface_original = pygame.surfarray.make_surface(img_rgb_original.swapaxes(0, 1))
    superficie_pygame.blit(surface_original, (0, 0))
    
    surface_roi_orig = pygame.surfarray.make_surface(roi_original_rgb.swapaxes(0, 1))
    superficie_pygame.blit(surface_roi_orig, (800, 0))
    
    surface_filtrada = pygame.surfarray.make_surface(imagen_filtrada_rgb.swapaxes(0, 1))
    superficie_pygame.blit(surface_filtrada, (1600, 0))

def main():
    pygame.init()
    # 2400x600 para ver las 3 vistas en paralelo
    pantalla = pygame.display.set_mode((2400, 600))
    pygame.display.set_caption("CARLA - 1. Original | 2. ROI Color | 3. ROI + Thresholding")

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
            
        path_file = os.path.join(path_dir, 'thresholding_times.txt')
        np.savetxt(path_file, historial_tiempos)

        if world is not None and original_settings is not None:
            world.apply_settings(original_settings)
        if camara_frontal is not None: camara_frontal.destroy()
        if vehiculo is not None: vehiculo.destroy()
        pygame.quit()

if __name__ == '__main__':
    main()