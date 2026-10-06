import carla
import random
import pygame
import numpy as np
import cv2

def procesar_camara_frontal(imagen, superficie_pygame):
    # 1. Extraer la imagen original bruta
    array = np.frombuffer(imagen.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (imagen.height, imagen.width, 4))
    img_bgr = array[:, :, :3]
    img_rgb_original = img_bgr[:, :, ::-1] 
    
    alto, ancho = img_bgr.shape[:2] # 600x800
    
    # ================= 2. DEFINIR LA REGIÓN DE INTERÉS (ROI) AMPLIADA =================
    # Mitad inferior de la cámara
    vertices_roi = np.array([[
        (0, alto),         # Abajo izquierda
        (0, 300),        # Arriba izquierda
        (800, 300),        # Arriba derecha
        (800, alto)        # Abajo derecha
    ]], dtype=np.int32)
    
    # Crear la máscara (1 canal de grises) y rellenar el polígono
    mascara_roi = np.zeros((alto, ancho), dtype=np.uint8)
    cv2.fillPoly(mascara_roi, vertices_roi, 255)
    
    # Aplicar la máscara a la imagen original (solo recortar, sin alterar colores)
    # bitwise_and con una máscara respeta los colores donde la máscara es 255
    roi_original_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=mascara_roi)
    roi_original_rgb = roi_original_bgr[:, :, ::-1]
    
    # ================= 3. APLICAR THRESHOLDING (Filtro Blanco) =================
    img_hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    blanco_bajo = np.array([0, 200, 0])
    blanco_alto = np.array([255, 255, 255])
    
    mascara_blanca = cv2.inRange(img_hls, blanco_bajo, blanco_alto)
    
    # Cruzamos el filtro de color con nuestra ROI para limpiar el cielo y edificios
    mascara_filtrada_roi = cv2.bitwise_and(mascara_blanca, mascara_roi)
    
    # Apilamos en 3 canales para Pygame
    imagen_filtrada_rgb = np.dstack((mascara_filtrada_roi, mascara_filtrada_roi, mascara_filtrada_roi)) 
    
    # ================= 4. RENDERIZAR EN 3 PANTALLAS =================
    # Izquierda (X=0): Original
    surface_original = pygame.surfarray.make_surface(img_rgb_original.swapaxes(0, 1))
    superficie_pygame.blit(surface_original, (0, 0))
    
    # Centro (X=800): Imagen a color recortada por la ROI
    surface_roi_orig = pygame.surfarray.make_surface(roi_original_rgb.swapaxes(0, 1))
    superficie_pygame.blit(surface_roi_orig, (800, 0))
    
    # Derecha (X=1600): Filtro blanco recortado por la ROI
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
        world = client.get_world()
        
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
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            pygame.display.flip()

    except KeyboardInterrupt:
        pass
    finally:
        if world is not None and original_settings is not None:
            world.apply_settings(original_settings)
        if camara_frontal is not None: camara_frontal.destroy()
        if vehiculo is not None: vehiculo.destroy()
        pygame.quit()

if __name__ == '__main__':
    main()