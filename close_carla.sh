#!/bin/bash

echo "Deteniendo contenedores de CARLA activos..."
# Detener el contenedor basado en el nombre de la imagen
docker stop $(docker ps -q --filter ancestor=carlasim/carla:0.9.16) 2>/dev/null

echo "Limpiando procesos zombies de Unreal Engine en el host..."
# Forzar el cierre de procesos de renderizado huérfanos
sudo killall -9 CarlaUE4-Linux-Shipping 2>/dev/null

echo "Reiniciando el servicio de Docker para liberar mapeos de red..."
# Limpieza profunda de sockets y puertos bloqueados
sudo systemctl restart docker

echo "¡Hecho! Puertos 2000 y 2001 liberados y limpios."
