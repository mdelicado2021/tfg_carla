# #!/bin/bash

# # Dar permisos a la interfaz gráfica local
# xhost +local:root

# echo "Lanzando CARLA 0.9.16 en modo Display (Low VRAM)..."

# # Lanzar el contenedor oficial con optimizaciones
# docker run -it \
#     --runtime=nvidia \
#     --net=host \
#     --user=$(id -u):$(id -g) \
#     --env=DISPLAY=$DISPLAY \
#     --env=NVIDIA_VISIBLE_DEVICES=all \
#     --env=NVIDIA_DRIVER_CAPABILITIES=all \
#     --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
#     carlasim/carla:0.9.16 bash CarlaUE4.sh -nosound -quality-level=Low -benchmark -fps=30


#!/bin/bash

echo "Lanzando CARLA 0.9.16 en modo OffScreen con Renderizado Vulkan (Optimizado)..."

docker run -it \
    --runtime=nvidia \
    --net=host \
    --user=$(id -u):$(id -g) \
    --env=DISPLAY=$DISPLAY \
    --env=NVIDIA_VISIBLE_DEVICES=all \
    --env=NVIDIA_DRIVER_CAPABILITIES=all \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    carlasim/carla:0.9.16 bash CarlaUE4.sh -vulkan -RenderOffScreen -nosound -quality-level=Low -benchmark -fps=30