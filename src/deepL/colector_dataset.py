import carla
import random
import os

def main():
    # Asegurar que los directorios existen
    os.makedirs('dataset/rgb', exist_ok=True)
    os.makedirs('dataset/semantic', exist_ok=True)

    vehiculo = None
    cam_rgb = None
    cam_sem = None
    original_settings = None

    try:
        print("Conectando con CARLA...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True 
        settings.fixed_delta_seconds = 0.1 # 10 FPS es suficiente para recolectar datos
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        
        blueprint_library = world.get_blueprint_library()
        modelo_coche = blueprint_library.filter('model3')[0]
        
        vehiculo = world.spawn_actor(modelo_coche, random.choice(world.get_map().get_spawn_points()))
        vehiculo.set_autopilot(True, traffic_manager.get_port())
        
        # 1. Configurar Cámara RGB (Input de la red neuronal)
        bp_rgb = blueprint_library.find('sensor.camera.rgb')
        bp_rgb.set_attribute('image_size_x', '800')
        bp_rgb.set_attribute('image_size_y', '600')
        
        # 2. Configurar Cámara Semántica (Ground Truth / Máscara ideal)
        bp_sem = blueprint_library.find('sensor.camera.semantic_segmentation')
        bp_sem.set_attribute('image_size_x', '800')
        bp_sem.set_attribute('image_size_y', '600')
        
        # Ambas cámaras en la EXACTA misma posición (capó del coche)
        transform = carla.Transform(carla.Location(x=2.0, z=1.5), carla.Rotation(pitch=-15))
        
        cam_rgb = world.spawn_actor(bp_rgb, transform, attach_to=vehiculo)
        cam_sem = world.spawn_actor(bp_sem, transform, attach_to=vehiculo)
        
        # Guardar en disco (usamos %06d para que los nombres sean 000001.png, 000002.png...)
        cam_rgb.listen(lambda image: image.save_to_disk('dataset/rgb/%06d.png' % image.frame))
        
        # ColorConverter.CityScapesPalette traduce las etiquetas matemáticas a colores visibles
        cam_sem.listen(lambda image: image.save_to_disk('dataset/semantic/%06d.png' % image.frame, carla.ColorConverter.CityScapesPalette))
        
        print("Grabando dataset... Deja que el coche conduzca unos 20 segundos.")
        print("Pulsa Ctrl+C en la terminal para detener la grabación.")
        
        # Bucle de simulación sin Pygame, solo avanza el reloj para generar fotogramas
        while True:
            world.tick() 

    except KeyboardInterrupt:
        print("\nFinalizando grabación y limpiando actores...")
    finally:
        if world is not None and original_settings is not None:
            world.apply_settings(original_settings)
        if cam_rgb is not None: cam_rgb.destroy()
        if cam_sem is not None: cam_sem.destroy()
        if vehiculo is not None: vehiculo.destroy()
        print("¡Dataset guardado con éxito!")

if __name__ == '__main__':
    main()