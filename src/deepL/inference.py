import carla
import random
import pygame
import numpy as np
import cv2
import torch
import sys
import os

# Asegurar que Python encuentre el archivo unet_model.py en el mismo directorio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from unet_model import UNet

# ================= 1. CARGA DEL MODELO =================
device = torch.device('cpu') 
print("Cargando modelo neuronal en memoria...")

# Instanciamos la arquitectura en blanco y le inyectamos los pesos del modelo entrenado
modelo = UNet(n_channels=3, n_classes=1).to(device)
ruta_pesos = 'saved_models/unet_carla_carriles.pth'
modelo.load_state_dict(torch.load(ruta_pesos, map_location=device, weights_only=True))

# Modo evaluación: desactiva el cálculo de gradientes y fija las capas BatchNorm
modelo.eval() 

def procesar_camara_frontal(imagen, superficie_pygame):
    array = np.frombuffer(imagen.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (imagen.height, imagen.width, 4))
    img_bgr = array[:, :, :3]
    img_rgb = img_bgr[:, :, ::-1].copy()

    # Redimensionamos la imagen para que la procese la red neuronal (inferencia rápida en CPU)
    img_rgb_small = cv2.resize(img_rgb, (400, 300))
    img_tensor = torch.from_numpy(img_rgb_small.transpose((2, 0, 1))).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        salida = modelo(img_tensor)
        mascara_binaria = torch.sigmoid(salida) > 0.5
        mascara_np_small = mascara_binaria.squeeze().cpu().numpy().astype(np.uint8)

    # Volvemos a escalar la máscara a su tamaño original de 800x600
    mascara_np = cv2.resize(mascara_np_small, (800, 600), interpolation=cv2.INTER_NEAREST)

    lienzo_prediccion = np.zeros_like(img_rgb)
    lienzo_prediccion[mascara_np == 1] = [0, 255, 0]

    resultado_final = cv2.addWeighted(img_rgb, 1, lienzo_prediccion, 0.4, 0)

    superficie_pygame.blit(pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1)), (0, 0))
    superficie_pygame.blit(pygame.surfarray.make_surface(resultado_final.swapaxes(0, 1)), (800, 0))

def main():
    pygame.init()
    pantalla = pygame.display.set_mode((1600, 600))
    pygame.display.set_caption("CARLA - Inferencia U-Net en Tiempo Real")

    vehiculo = None
    camara_frontal = None
    original_settings = None

    try:
        print("Conectando con CARLA...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(60.0)
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
        
        print("Rodando red neuronal. Pulsa Ctrl+C para salir.")
        
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