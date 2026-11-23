import os
import queue
import random
import carla
import cv2
import numpy as np

NUM_FRAMES = 150
SALTOS_TICK = 10  # Guarda 1 imagen cada 10 ticks (0.5 s) para tener variedad de carretera
DIR_RGB = 'eval_dataset/rgb'
DIR_SEM = 'eval_dataset/semantic'

def main():
    os.makedirs(DIR_RGB, exist_ok=True)
    os.makedirs(DIR_SEM, exist_ok=True)

    vehiculo = None
    cam_rgb = None
    cam_sem = None
    world = None
    original_settings = None

    try:
        print("Conectando con CARLA...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(60.0)
        world = client.get_world()

        if not world.get_map().name.endswith('Town04'):
            print("Cargando Town04...")
            world = client.load_world('Town04')

        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)

        bp_lib = world.get_blueprint_library()
        vehiculo = world.spawn_actor(bp_lib.filter('model3')[0], random.choice(world.get_map().get_spawn_points()))
        vehiculo.set_autopilot(True, tm.get_port())

        transform = carla.Transform(carla.Location(x=2.0, z=1.5), carla.Rotation(pitch=-15))

        bp_rgb = bp_lib.find('sensor.camera.rgb')
        bp_rgb.set_attribute('image_size_x', '800')
        bp_rgb.set_attribute('image_size_y', '600')
        bp_rgb.set_attribute('fov', '90')
        cam_rgb = world.spawn_actor(bp_rgb, transform, attach_to=vehiculo)

        bp_sem = bp_lib.find('sensor.camera.semantic_segmentation')
        bp_sem.set_attribute('image_size_x', '800')
        bp_sem.set_attribute('image_size_y', '600')
        bp_sem.set_attribute('fov', '90')
        cam_sem = world.spawn_actor(bp_sem, transform, attach_to=vehiculo)

        q_rgb = queue.Queue()
        q_sem = queue.Queue()
        cam_rgb.listen(q_rgb.put)
        cam_sem.listen(q_sem.put)

        # Warm-up inicial de 20 ticks para que el coche arranque
        for _ in range(20):
            world.tick()
            q_rgb.get()
            q_sem.get()

        guardadas = 0
        tick_count = 0
        print(f"Capturando {NUM_FRAMES} pares sincronizados en eval_dataset/...")

        while guardadas < NUM_FRAMES:
            world.tick()
            img_rgb_raw = q_rgb.get()
            img_sem_raw = q_sem.get()
            tick_count += 1

            if tick_count % SALTOS_TICK != 0:
                continue

            if img_rgb_raw.frame == img_sem_raw.frame:
                # RGB a BGR para OpenCV
                arr_rgb = np.frombuffer(img_rgb_raw.raw_data, dtype=np.uint8)
                arr_rgb = arr_rgb.reshape((600, 800, 4))[:, :, :3]

                # Semántica con paleta CityScapes (igual que en dataset_loader.py)
                img_sem_raw.convert(carla.ColorConverter.CityScapesPalette)
                arr_sem = np.frombuffer(img_sem_raw.raw_data, dtype=np.uint8)
                arr_sem = arr_sem.reshape((600, 800, 4))[:, :, :3]

                guardadas += 1
                nombre = f"{guardadas:06d}.png"
                cv2.imwrite(os.path.join(DIR_RGB, nombre), arr_rgb)
                cv2.imwrite(os.path.join(DIR_SEM, nombre), arr_sem)

                if guardadas % 25 == 0 or guardadas == NUM_FRAMES:
                    print(f"  Progreso: {guardadas}/{NUM_FRAMES} imágenes guardadas")

        print("Captura completada con éxito.")

    finally:
        if cam_rgb is not None:
            cam_rgb.stop()
            cam_rgb.destroy()
        if cam_sem is not None:
            cam_sem.stop()
            cam_sem.destroy()
        if vehiculo is not None:
            vehiculo.destroy()
        if world is not None and original_settings is not None:
            world.tick()
            world.apply_settings(original_settings)

if __name__ == '__main__':
    main()