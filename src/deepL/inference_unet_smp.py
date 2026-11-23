import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import carla
import random
import pygame
import numpy as np
import cv2
import torch
import sys
import math
import queue
import rclpy
from sensor_msgs.msg import PointCloud2, PointField
import signal
import time
import segmentation_models_pytorch as smp

historial_tiempos = []

def empaquetar_pointcloud2(puntos_3d, frame_id="carla_camera"):
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = len(puntos_3d)
    
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    
    msg.data = np.asarray(puntos_3d, dtype=np.float32).tobytes()
    return msg

def construir_matriz_intrinseca(width, height, fov):
    k = np.identity(3)
    f = width / (2.0 * math.tan(fov * math.pi / 360.0))
    cx = width / 2.0
    cy = height / 2.0
    
    k[0, 0] = f
    k[1, 1] = f
    k[0, 2] = cx
    k[1, 2] = cy
    return k

# ================= 1. CARGA DEL MODELO =================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
etiqueta_dev = "GPU" if device.type == 'cuda' else "CPU"
print(f"Cargando modelo U-Net SMP (ResNet-18) en {etiqueta_dev}...")

modelo = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=3,
    classes=1
).to(device)

ruta_pesos = 'saved_models/unet_smp_resnet18.pth'
modelo.load_state_dict(torch.load(ruta_pesos, map_location=device, weights_only=True))
modelo.eval()


def procesar_camara(img_rgb, profundidad_metros, K):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_inicio = time.perf_counter()

    # --- 1. PROCESAMIENTO RGB (384x288, múltiplos de 32) ---
    img_rgb_small = cv2.resize(img_rgb, (384, 288))
    img_tensor = torch.from_numpy(img_rgb_small.transpose((2, 0, 1))).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
        salida = modelo(img_tensor)
        mascara_binaria = torch.sigmoid(salida) > 0.5
        mascara_np_small = mascara_binaria.squeeze().cpu().numpy().astype(np.uint8)

    mascara_np = cv2.resize(mascara_np_small, (800, 600), interpolation=cv2.INTER_NEAREST)

    # --- 2. PROYECCIÓN 3D ---
    v, u = np.where(mascara_np == 1)
    puntos_3d = []
    
    if len(u) > 0:
        Z = profundidad_metros[v, u]
        mascara_distancia = (Z > 1.0) & (Z < 50.0)
        u_validos = u[mascara_distancia]
        v_validos = v[mascara_distancia]
        Z_validos = Z[mascara_distancia]
        
        X = (u_validos - K[0, 2]) * Z_validos / K[0, 0]
        Y = (v_validos - K[1, 2]) * Z_validos / K[1, 1]

        puntos_3d = np.column_stack((Z_validos, -X, -Y))

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_fin = time.perf_counter()

    tiempo_ms = (t_fin - t_inicio) * 1000.0
    historial_tiempos.append(tiempo_ms)
    print(f"Latencia Percepción (U-Net SMP {etiqueta_dev}): {tiempo_ms:.2f} ms")

    # --- 3. DIBUJADO PARA PYGAME ---
    lienzo_prediccion = np.zeros_like(img_rgb)
    lienzo_prediccion[mascara_np == 1] = [0, 255, 0]
    resultado_final = cv2.addWeighted(img_rgb, 1, lienzo_prediccion, 0.4, 0)

    return puntos_3d, resultado_final


def callback_rgb(imagen_raw, cola):
    array = np.frombuffer(imagen_raw.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (imagen_raw.height, imagen_raw.width, 4))
    img_rgb = array[:, :, :3][:, :, ::-1].copy()
    cola.put((imagen_raw.frame, img_rgb))

def callback_depth(imagen_raw, cola):
    array = np.frombuffer(imagen_raw.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (imagen_raw.height, imagen_raw.width, 4))
    B = array[:, :, 0].astype(np.float32)
    G = array[:, :, 1].astype(np.float32)
    R = array[:, :, 2].astype(np.float32)
    normalized = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0)
    profundidad_metros = 1000.0 * normalized
    cola.put((imagen_raw.frame, profundidad_metros))

def main():
    pygame.init()
    pantalla = pygame.display.set_mode((1600, 600))
    pygame.display.set_caption("CARLA - Inferencia U-Net SMP (ResNet-18) 3D y ROS2")

    rclpy.init()
    nodo_ros = rclpy.create_node('carla_lane_detector_smp')
    pub_nube = nodo_ros.create_publisher(PointCloud2, '/carla/carriles_3d', 10)

    vehiculo = None
    camara_frontal = None
    camara_depth = None
    original_settings = None
    world = None

    try:
        print("Conectando con CARLA...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(60.0)
        map_name = 'Town04'
        world = client.get_world()

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
        bp_camara.set_attribute('fov', '90')
        
        transform_frontal = carla.Transform(carla.Location(x=2.0, z=1.5), carla.Rotation(pitch=-15))
        camara_frontal = world.spawn_actor(bp_camara, transform_frontal, attach_to=vehiculo)
        
        bp_depth = blueprint_library.find('sensor.camera.depth')
        bp_depth.set_attribute('image_size_x', '800')
        bp_depth.set_attribute('image_size_y', '600')
        bp_depth.set_attribute('fov', '90')
        
        camara_depth = world.spawn_actor(bp_depth, transform_frontal, attach_to=vehiculo)
        
        matriz_K = construir_matriz_intrinseca(800, 600, 90)
        
        image_queue_rgb = queue.Queue()
        image_queue_depth = queue.Queue()
        
        camara_frontal.listen(lambda image: callback_rgb(image, image_queue_rgb))
        camara_depth.listen(lambda image: callback_depth(image, image_queue_depth))
        
        def manejador_senales(sig, frame):
            print("\n[!] Apagando sistema de forma segura para proteger CARLA...")
            nonlocal running
            running = False

        signal.signal(signal.SIGINT, manejador_senales)
        
        print("Nodo ROS2 activo. Nube publicándose en /carla/carriles_3d. Pulsa Ctrl+C para salir.")
        
        running = True
        while running:
            world.tick()
            
            if len(historial_tiempos) >= 100:
                print("Límite de 100 fotogramas alcanzado. Guardando resultados...")
                running = False
            
            frame_rgb, img_rgb = image_queue_rgb.get()
            frame_depth, profundidad_metros = image_queue_depth.get()
            
            if frame_rgb == frame_depth:
                puntos_3d, resultado_final = procesar_camara(img_rgb, profundidad_metros, matriz_K)

                if puntos_3d is not None and len(puntos_3d) > 0:
                    msg_pc = empaquetar_pointcloud2(puntos_3d)
                    pub_nube.publish(msg_pc)

                pantalla.blit(pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1)), (0, 0))
                pantalla.blit(pygame.surfarray.make_surface(resultado_final.swapaxes(0, 1)), (800, 0))

            rclpy.spin_once(nodo_ros, timeout_sec=0)

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
            
        sufijo = 'gpu' if device.type == 'cuda' else 'cpu'
        path_file = os.path.join(path_dir, f'unet_smp_{sufijo}_times.txt')
        np.savetxt(path_file, historial_tiempos)

        print("Limpiando entorno y restaurando simulador...")
        if camara_frontal is not None: 
            camara_frontal.stop()
            camara_frontal.destroy()
        if camara_depth is not None: 
            camara_depth.stop()
            camara_depth.destroy()
        if vehiculo is not None: 
            vehiculo.destroy()
            
        if world is not None and original_settings is not None:
            world.tick()
            world.apply_settings(original_settings)
        
        pygame.quit()
        if rclpy.ok():
            nodo_ros.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()