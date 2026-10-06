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
    
    # ================= PASO 1: REGIÓN DE INTERÉS (ROI) =================
    vertices_roi = np.array([[
        (0, alto),         
        (0, 300),        
        (800, 300),        
        (800, alto)        
    ]], dtype=np.int32)
    
    mascara_roi = np.zeros((alto, ancho), dtype=np.uint8)
    cv2.fillPoly(mascara_roi, vertices_roi, 255)
    
    roi_original_rgb = cv2.bitwise_and(img_rgb_original, img_rgb_original, mask=mascara_roi)
    
    # ================= PASO 2: THRESHOLDING (Filtro de Color) =================
    img_hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    blanco_bajo = np.array([0, 200, 0])
    blanco_alto = np.array([255, 255, 255])
    
    mascara_blanca = cv2.inRange(img_hls, blanco_bajo, blanco_alto)
    mascara_filtrada_roi = cv2.bitwise_and(mascara_blanca, mascara_roi)
    
    imagen_thresholding = np.dstack((mascara_filtrada_roi, mascara_filtrada_roi, mascara_filtrada_roi))
    
    # ================= PASO 3: CANNY (Detección de Bordes) =================
    bordes = cv2.Canny(mascara_filtrada_roi, 50, 150)
    imagen_canny = np.dstack((bordes, bordes, bordes))
    
    # ================= PASO 4: TRANSFORMADA DE HOUGH =================
    # Hacemos una copia de la imagen original para dibujar las rectas matemáticas encima
    imagen_hough = np.copy(img_rgb_original)
    
    # Parámetros: imagen Canny, resolución rho (1 px), resolución theta (1 grado),
    # umbral de votos (50), longitud mínima de recta (40px), hueco máximo entre rectas (20px)
    lineas = cv2.HoughLinesP(bordes, 1, np.pi/180, 50, minLineLength=40, maxLineGap=20)
    
    # Si encuentra líneas, las iteramos y las dibujamos en rojo
    if lineas is not None:
            for linea in lineas:
                # .flatten() aplana la estructura sin importar los corchetes que tenga
                x1, y1, x2, y2 = linea.flatten()
                
                # Forzamos int() por seguridad para que cv2.line no falle con los tipos
                cv2.line(imagen_hough, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 4)
                
    # ================= RENDERIZAR EN CUADRÍCULA 3x2 =================
    # Fila 1: Original | ROI | Thresholding
    superficie_pygame.blit(pygame.surfarray.make_surface(img_rgb_original.swapaxes(0, 1)), (0, 0))
    superficie_pygame.blit(pygame.surfarray.make_surface(roi_original_rgb.swapaxes(0, 1)), (800, 0))
    superficie_pygame.blit(pygame.surfarray.make_surface(imagen_thresholding.swapaxes(0, 1)), (1600, 0))
    
    # Fila 2: Canny | Hough | (Panel Negro vacío)
    superficie_pygame.blit(pygame.surfarray.make_surface(imagen_canny.swapaxes(0, 1)), (0, 600))
    superficie_pygame.blit(pygame.surfarray.make_surface(imagen_hough.swapaxes(0, 1)), (800, 600))
    pygame.draw.rect(superficie_pygame, (0,0,0), (1600, 600, 800, 600))

def main():
    pygame.init()
    # Ampliamos a 2400x1200 para dar cabida a 3 columnas y 2 filas
    pantalla = pygame.display.set_mode((2400, 1200))
    pygame.display.set_caption("CARLA - Pipeline 3x2: Original | ROI | Threshold | Canny | Hough")

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