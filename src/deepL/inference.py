import carla
import random
import pygame
import numpy as np
import cv2
import torch
import sys
import os
import math
import queue
import rclpy
from sensor_msgs.msg import PointCloud2, PointField
import signal

def empaquetar_pointcloud2(puntos_3d, frame_id="carla_camera"):
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = len(puntos_3d)
    
    # Definimos la estructura de la nube: Coordenadas flotantes de 32 bits
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
    ]
    msg.is_bigendian = False
    msg.point_step = 12 # 4 bytes por flotante * 3 coordenadas
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    
    # Empaquetamos la matriz de NumPy en bytes crudos
    msg.data = np.asarray(puntos_3d, dtype=np.float32).tobytes()
    return msg

def construir_matriz_intrinseca(width, height, fov):
    """
    Construye la matriz intrínseca K de la cámara.
    (Ideal para documentar en la memoria del TFG)
    """
    k = np.identity(3)
    # Convertir el FOV de grados a radianes y calcular la distancia focal (f)
    f = width / (2.0 * math.tan(fov * math.pi / 360.0))
    
    # Centro óptico de la imagen (cx, cy)
    cx = width / 2.0
    cy = height / 2.0
    
    k[0, 0] = f   # fx
    k[1, 1] = f   # fy
    k[0, 2] = cx
    k[1, 2] = cy
    return k

def decodificar_profundidad(imagen_depth):
    """
    Convierte el mapa de profundidad de CARLA (codificado en RGB) a metros reales.
    """
    # Convertimos los datos brutos a array BGR
    array = np.frombuffer(imagen_depth.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (imagen_depth.height, imagen_depth.width, 4))
    
    # Separamos los canales (B, G, R) - ignoramos el Alpha
    B = array[:, :, 0].astype(np.float32)
    G = array[:, :, 1].astype(np.float32)
    R = array[:, :, 2].astype(np.float32)
    
    # Fórmula oficial de CARLA para decodificar profundidad en metros (máximo 1000m)
    normalized = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0)
    profundidad_metros = 1000.0 * normalized
    
    return profundidad_metros

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


def procesar_camara(img_rgb, profundidad_metros, K):
    # --- 1. PROCESAMIENTO RGB ---
    # Ya recibimos img_rgb como NumPy array desde el callback
    img_rgb_small = cv2.resize(img_rgb, (400, 300))
    img_tensor = torch.from_numpy(img_rgb_small.transpose((2, 0, 1))).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        salida = modelo(img_tensor)
        mascara_binaria = torch.sigmoid(salida) > 0.5
        mascara_np_small = mascara_binaria.squeeze().cpu().numpy().astype(np.uint8)

    mascara_np = cv2.resize(mascara_np_small, (800, 600), interpolation=cv2.INTER_NEAREST)

    # --- 2. PROYECCIÓN 3D ---
    v, u = np.where(mascara_np == 1)
    puntos_3d = [] # Inicializamos vacío por si el modelo no detecta líneas en este frame
    
    if len(u) > 0:
        Z = profundidad_metros[v, u]
        mascara_distancia = (Z > 1.0) & (Z < 50.0)
        u_validos = u[mascara_distancia]
        v_validos = v[mascara_distancia]
        Z_validos = Z[mascara_distancia]
        
        X = (u_validos - K[0, 2]) * Z_validos / K[0, 0]
        Y = (v_validos - K[1, 2]) * Z_validos / K[1, 1]

        # Adaptación de cámara a ROS2
        puntos_3d = np.column_stack((Z_validos, -X, -Y))

    # --- 3. DIBUJADO PARA PYGAME ---
    lienzo_prediccion = np.zeros_like(img_rgb)
    lienzo_prediccion[mascara_np == 1] = [0, 255, 0]
    resultado_final = cv2.addWeighted(img_rgb, 1, lienzo_prediccion, 0.4, 0)

    # Retornamos las variables para que el bucle principal las publique y dibuje
    return puntos_3d, resultado_final


def callback_rgb(imagen_raw, cola):
    # Extraemos la matriz e independizamos la memoria
    array = np.frombuffer(imagen_raw.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (imagen_raw.height, imagen_raw.width, 4))
    img_rgb = array[:, :, :3][:, :, ::-1].copy()
    
    # Solo metemos a la cola el ID del fotograma y la matriz NumPy
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
    # 1. Inicialización de Pygame
    pygame.init()
    pantalla = pygame.display.set_mode((1600, 600))
    pygame.display.set_caption("CARLA - Inferencia U-Net 3D y ROS2")

    # 2. Inicialización de ROS 2
    rclpy.init()
    nodo_ros = rclpy.create_node('carla_lane_detector')
    pub_nube = nodo_ros.create_publisher(PointCloud2, '/carla/carriles_3d', 10)

    vehiculo = None
    camara_frontal = None
    camara_depth = None
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
        
        # --- CÁMARA RGB ---
        bp_camara = blueprint_library.find('sensor.camera.rgb')
        bp_camara.set_attribute('image_size_x', '800')
        bp_camara.set_attribute('image_size_y', '600')
        bp_camara.set_attribute('fov', '90')
        
        transform_frontal = carla.Transform(carla.Location(x=2.0, z=1.5), carla.Rotation(pitch=-15))
        camara_frontal = world.spawn_actor(bp_camara, transform_frontal, attach_to=vehiculo)
        
        # --- CÁMARA DE PROFUNDIDAD ---
        bp_depth = blueprint_library.find('sensor.camera.depth')
        bp_depth.set_attribute('image_size_x', '800')
        bp_depth.set_attribute('image_size_y', '600')
        bp_depth.set_attribute('fov', '90')
        
        camara_depth = world.spawn_actor(bp_depth, transform_frontal, attach_to=vehiculo)
        
        matriz_K = construir_matriz_intrinseca(800, 600, 90)
        
        # --- BUZONES (COLAS) DE EXTRACCIÓN SEGURA ---
        image_queue_rgb = queue.Queue()
        image_queue_depth = queue.Queue()
        
        # Usamos las funciones de callback para que destruyan el objeto C++ y liberen RAM
        camara_frontal.listen(lambda image: callback_rgb(image, image_queue_rgb))
        camara_depth.listen(lambda image: callback_depth(image, image_queue_depth))
        
        # NUEVO INTERCEPTOR DE CTRL+C
        def manejador_senales(sig, frame):
            print("\n[!] Apagando sistema de forma segura para proteger CARLA...")
            nonlocal running
            running = False

        signal.signal(signal.SIGINT, manejador_senales)
        
        print("Nodo ROS2 activo. Nube publicándose en /carla/carriles_3d. Pulsa Ctrl+C para salir.")
        
        running = True
        while running:
            world.tick() 
            
            # Sacamos las matrices independientes de cada buzón
            frame_rgb, img_rgb = image_queue_rgb.get()
            frame_depth, profundidad_metros = image_queue_depth.get()
            
            # Comprobación de sincronía: procesamos solo si son del mismo instante
            if frame_rgb == frame_depth:
                # Llama a la función solo con la imagen, la profundidad y la matriz K
                puntos_3d, resultado_final = procesar_camara(img_rgb, profundidad_metros, matriz_K)

                # Publicar a ROS 2 / RViz
                if puntos_3d is not None and len(puntos_3d) > 0:
                    msg_pc = empaquetar_pointcloud2(puntos_3d)
                    pub_nube.publish(msg_pc)

                # Renderizado local en Pygame
                pantalla.blit(pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1)), (0, 0))
                pantalla.blit(pygame.surfarray.make_surface(resultado_final.swapaxes(0, 1)), (800, 0))

            # Mantiene vivo el ecosistema ROS 2 para que los mensajes fluyan a RViz
            rclpy.spin_once(nodo_ros, timeout_sec=0)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            pygame.display.flip()

    except KeyboardInterrupt:
        pass
    finally:
        print("Limpiando entorno y restaurando simulador...")
        # 1. Apagar y destruir sensores
        if camara_frontal is not None: 
            camara_frontal.stop()
            camara_frontal.destroy()
        if camara_depth is not None: 
            camara_depth.stop()
            camara_depth.destroy()
        if vehiculo is not None: 
            vehiculo.destroy()
            
        # 2. Restaurar el mundo de forma segura
        if world is not None and original_settings is not None:
            world.tick() # Vital: Registra la destrucción en el servidor
            world.apply_settings(original_settings)
        
        # 3. Cierre de librerías
        pygame.quit()
        if rclpy.ok():
            nodo_ros.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
