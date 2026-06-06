#!/usr/bin/env python3
import carla
import random
import time

def main():
    try:
        # 1. Conectar con el servidor Docker de CARLA
        print("Conectando con el servidor CARLA (Puerto 2000)...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0) # Espera máxima de conexión
        
        world = client.get_world()
        print("¡Conectado con éxito al mapa pregrabado!")
        
        # 3. El Blueprint Library es el "catálogo" de objetos de CARLA
        blueprint_library = world.get_blueprint_library()
        
        # 4. Elegir un coche (por ejemplo, un Tesla Model 3)
        modelo_coche = blueprint_library.filter('model3')[0]
        
        # 5. Buscar un punto de aparición (spawn point) libre en el mapa
        puntos_spawn = world.get_map().get_spawn_points()
        punto_inicio = random.choice(puntos_spawn)
        
        # 6. Spamear (crear) el coche en el simulador
        print("Spawnando vehículo en el mapa...")
        vehiculo = world.spawn_actor(modelo_coche, punto_inicio)
        
        # 7. Activar el piloto automático de CARLA para que ruede solo
        vehiculo.set_autopilot(True)
        print("¡Coche creado con piloto automático activo!")
        
        # 8. Bucle para leer datos en tiempo real por consola
        print("Imprimiendo telemetría. Pulsa Ctrl+C para terminar...")
        while True:
            # Obtener la posición actual del coche en el mundo
            transform = vehiculo.get_transform()
            print(f"Posición coche -> X: {transform.location.x:.2f}, Y: {transform.location.y:.2f}, Z: {transform.location.z:.2f}", end='\r')
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nTerminando simulación por el usuario...")
    finally:
        # 9. Limpieza obligatoria: Al cerrar el script, destruimos el coche del Docker
        if 'vehiculo' in locals():
            print("Destruyendo vehículo para liberar memoria...")
            vehiculo.destroy()
        print("Simulación finalizada limpiamente.")

if __name__ == '__main__':
    main()