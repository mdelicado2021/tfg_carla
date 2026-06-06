#!/usr/bin/env python3
import carla
import random
import time
import pygame
import numpy as np

def procesar_imagen(imagen, superficie_pygame):
    array = np.frombuffer(imagen.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (imagen.height, imagen.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    superficie_pygame.blit(surface, (0, 0))

def main():
    pygame.init()
    pantalla = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("CARLA - Modo Sincrónico Perfecto")
    clock = pygame.time.Clock()

    vehiculo = None
    camara = None
    original_settings = None

    try:
        print("Conectando con el servidor CARLA...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        
        # =========================================================
        # 1. CONFIGURAR MODO SINCRÓNICO (Bloqueo de reloj)
        # =========================================================
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True # Activar modo síncrono
        settings.fixed_delta_seconds = 0.05 # Forzar a que cada frame dure exactamente 0.05s (20 FPS)
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True) # Sincronizar también el piloto automático
        traffic_manager.set_global_distance_to_leading_vehicle(3.0)
        
        # 2. Spawnear vehículo de forma segura
        blueprint_library = world.get_blueprint_library()
        modelo_coche = blueprint_library.filter('model3')[0]
        puntos_spawn = world.get_map().get_spawn_points()
        
        for i in range(10):
            punto_inicio = random.choice(puntos_spawn)
            vehiculo = world.try_spawn_actor(modelo_coche, punto_inicio)
            if vehiculo is not None:
                break
        
        if vehiculo is None:
            print("No se pudo crear el vehículo.")
            return

        vehiculo.set_autopilot(True, traffic_manager.get_port())
        print("¡Coche creado y sincronizado con el reloj del sistema!")
        
        # 3. Configurar cámara trasera
        bp_camara = blueprint_library.find('sensor.camera.rgb')
        bp_camara.set_attribute('image_size_x', '800')
        bp_camara.set_attribute('image_size_y', '600')
        transform_camara = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=-15))
        camara = world.spawn_actor(bp_camara, transform_camara, attach_to=vehiculo)
        camara.listen(lambda image: procesar_imagen(image, pantalla))
        
        print("Simulación en marcha. El coche debería conducir sobre raíles ahora...")
        
        running = True
        while running:
            # Mandar la señal al Docker de que avance un frame en las físicas
            world.tick() 
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            if vehiculo.is_alive:
                transform = vehiculo.get_transform()
                print(f"Posición coche -> X: {transform.location.x:.2f}, Y: {transform.location.y:.2f}, Z: {transform.location.z:.2f}", end='\r')
            else:
                running = False
            
            pygame.display.flip()
            # Ya no usamos clock.tick() porque el ritmo lo marca world.tick() de forma matemática

    except KeyboardInterrupt:
        print("\nTerminando simulación...")
    finally:
        # =========================================================
        # 4. RESTAURAR CONFIGURACIÓN AL SALIR (Muy importante)
        # =========================================================
        if world is not None and original_settings is not None:
            print("\nRestaurando modo asíncrono en el servidor...")
            world.apply_settings(original_settings)
            
        print("Limpiando actores...")
        if camara is not None and camara.is_alive:
            camara.destroy()
        if vehiculo is not None and vehiculo.is_alive:
            vehiculo.destroy()
        pygame.quit()
        print("Finalizado.")

if __name__ == '__main__':
    main()