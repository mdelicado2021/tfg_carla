import carla
import random
import pygame
import numpy as np
import cv2

# ================= MEMORIA TEMPORAL =================
fit_izq_previo = None
fit_der_previo = None
# ====================================================

def procesar_camara_frontal(imagen, superficie_pygame):
    global fit_izq_previo, fit_der_previo
    
    array = np.frombuffer(imagen.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (imagen.height, imagen.width, 4))
    img_bgr = array[:, :, :3]
    img_rgb_original = img_bgr[:, :, ::-1] 
    
    alto, ancho = img_bgr.shape[:2]
    
    # 1. MATRICES DE PERSPECTIVA
    src_pts = np.float32([[250, 350], [550, 350], [800, 500], [0, 500]])
    dst_pts = np.float32([[200, 0], [600, 0], [600, alto], [200, alto]])
    matriz_perspectiva = cv2.getPerspectiveTransform(src_pts, dst_pts)
    matriz_inversa = cv2.getPerspectiveTransform(dst_pts, src_pts)
    
    vertices_roi = np.array([src_pts], dtype=np.int32)
    mascara_roi = np.zeros((alto, ancho), dtype=np.uint8)
    cv2.fillPoly(mascara_roi, vertices_roi, 255)
    roi_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=mascara_roi)
    ipm_roi_bgr = cv2.warpPerspective(roi_bgr, matriz_perspectiva, (ancho, alto))

    # 2. MÁSCARA DUAL (BLANCO + AMARILLO)
    ipm_roi_hls = cv2.cvtColor(ipm_roi_bgr, cv2.COLOR_BGR2HLS)
    
    # Rango para pintura blanca
    mask_blanca = cv2.inRange(ipm_roi_hls, np.array([0, 200, 0]), np.array([255, 255, 255]))
    # Rango para pintura amarilla (Ajustado para iluminación de CARLA)
    mask_amarilla = cv2.inRange(ipm_roi_hls, np.array([15, 30, 115]), np.array([35, 204, 255]))
    
    # Combinamos ambas máscaras
    mask_ipm_roi = cv2.bitwise_or(mask_blanca, mask_amarilla)
    out_img = np.dstack((mask_ipm_roi, mask_ipm_roi, mask_ipm_roi))

    # 3. VENTANAS DESLIZANTES
    histograma = np.sum(mask_ipm_roi[alto//2:, :], axis=0)
    punto_medio = int(histograma.shape[0]//2)
    
    resultado_final = np.copy(img_rgb_original)
    
    # Valores por defecto para el ajuste actual
    fit_izq = None
    fit_der = None
    
    if np.max(histograma[:punto_medio]) > 0 or np.max(histograma[punto_medio:]) > 0:
        base_izq = np.argmax(histograma[:punto_medio])
        base_der = np.argmax(histograma[punto_medio:]) + punto_medio
        
        n_ventanas = 9
        alto_ventana = int(alto // n_ventanas)
        margen = 100
        minpix = 50
        
        nonzero = mask_ipm_roi.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
        
        centro_actual_izq = base_izq
        centro_actual_der = base_der
        indices_izq = []
        indices_der = []
        
        for ventana in range(n_ventanas):
            win_y_low = alto - (ventana + 1) * alto_ventana
            win_y_high = alto - ventana * alto_ventana
            
            # Ventana Izquierda
            win_xleft_low = centro_actual_izq - margen
            win_xleft_high = centro_actual_izq + margen
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            buenos_izq = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & 
                          (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
            indices_izq.append(buenos_izq)
            if len(buenos_izq) > minpix: centro_actual_izq = int(np.mean(nonzero_x[buenos_izq]))
                
            # Ventana Derecha
            win_xright_low = centro_actual_der - margen
            win_xright_high = centro_actual_der + margen
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            buenos_der = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & 
                          (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
            indices_der.append(buenos_der)
            if len(buenos_der) > minpix: centro_actual_der = int(np.mean(nonzero_x[buenos_der]))
                
        indices_izq = np.concatenate(indices_izq)
        indices_der = np.concatenate(indices_der)
        
        # Validación de suficiencia de píxeles (Evita colapsos de polyfit)
        if len(indices_izq) > 200:
            fit_izq = np.polyfit(nonzero_y[indices_izq], nonzero_x[indices_izq], 2)
            out_img[nonzero_y[indices_izq], nonzero_x[indices_izq]] = [255, 0, 0]
        
        if len(indices_der) > 200:
            fit_der = np.polyfit(nonzero_y[indices_der], nonzero_x[indices_der], 2)
            out_img[nonzero_y[indices_der], nonzero_x[indices_der]] = [0, 0, 255]

    # ================= 4. REGLAS MATEMÁTICAS DE RESPALDO =================
    ancho_carril_estimado = 400 # Distancia en píxeles entre líneas en IPM
    
    # Regla 1: Inferencia por paralelismo (si falta una, clonamos la otra)
    if fit_izq is not None and fit_der is None:
        fit_der = np.array([fit_izq[0], fit_izq[1], fit_izq[2] + ancho_carril_estimado])
    elif fit_der is not None and fit_izq is None:
        fit_izq = np.array([fit_der[0], fit_der[1], fit_der[2] - ancho_carril_estimado])
        
    # Regla 2: Memoria Temporal (si fallan ambas, tiramos de historial)
    if fit_izq is None: fit_izq = fit_izq_previo
    if fit_der is None: fit_der = fit_der_previo
    
    # Actualizamos el historial para el próximo fotograma
    if fit_izq is not None: fit_izq_previo = fit_izq
    if fit_der is not None: fit_der_previo = fit_der

    # ================= 5. PROYECCIÓN =================
    if fit_izq is not None and fit_der is not None:
        ploty = np.linspace(0, alto-1, alto)
        fit_x_izq = fit_izq[0]*ploty**2 + fit_izq[1]*ploty + fit_izq[2]
        fit_x_der = fit_der[0]*ploty**2 + fit_der[1]*ploty + fit_der[2]
        
        # Dibujar en el panel central
        pts_izq_center = np.array([np.transpose(np.vstack([fit_x_izq, ploty]))], np.int32)
        pts_der_center = np.array([np.transpose(np.vstack([fit_x_der, ploty]))], np.int32)
        cv2.polylines(out_img, pts_izq_center, False, (0, 255, 255), 4)
        cv2.polylines(out_img, pts_der_center, False, (0, 255, 255), 4)
        
        # Dibujar proyección de Realidad Aumentada
        lienzo_pajaro = np.zeros_like(img_bgr)
        pts_izq = np.array([np.transpose(np.vstack([fit_x_izq, ploty]))])
        pts_der = np.array([np.flipud(np.transpose(np.vstack([fit_x_der, ploty])))])
        pts_poligono = np.hstack((pts_izq, pts_der))
        
        cv2.fillPoly(lienzo_pajaro, np.int_([pts_poligono]), (0, 100, 0)) 
        cv2.polylines(lienzo_pajaro, np.int_([pts_izq]), False, (255, 0, 0), 20) 
        cv2.polylines(lienzo_pajaro, np.int_([pts_der]), False, (0, 0, 255), 20) 
        
        lienzo_original = cv2.warpPerspective(lienzo_pajaro, matriz_inversa, (ancho, alto))
        resultado_final = cv2.addWeighted(img_rgb_original, 1, lienzo_original, 0.5, 0)

    # 6. RENDERIZADO (3 Paneles)
    superficie_pygame.blit(pygame.surfarray.make_surface(img_rgb_original.swapaxes(0, 1)), (0, 0))
    superficie_pygame.blit(pygame.surfarray.make_surface(out_img.swapaxes(0, 1)), (800, 0))
    superficie_pygame.blit(pygame.surfarray.make_surface(resultado_final.swapaxes(0, 1)), (1600, 0))

def main():
    pygame.init()
    pantalla = pygame.display.set_mode((2400, 600))
    pygame.display.set_caption("CARLA - Ventanas Blindadas (Filtro Amarillo + Memoria + Paralelismo)")

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        
        settings = world.get_settings()
        settings.synchronous_mode = True 
        settings.fixed_delta_seconds = 0.05 
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        
        blueprint_library = world.get_blueprint_library()
        modelo_coche = blueprint_library.filter('model3')[0]
        
        vehiculo = world.spawn_actor(modelo_coche, random.choice(world.get_map().get_spawn_points()))
        vehiculo.set_autopilot(True, traffic_manager.get_port())
        
        bp_camara = blueprint_library.find('sensor.camera.rgb')
        bp_camara.set_attribute('image_size_x', '800')
        bp_camara.set_attribute('image_size_y', '600')
        
        transform_frontal = carla.Transform(carla.Location(x=2.0, z=1.5), carla.Rotation(pitch=-15))
        camara_frontal = world.spawn_actor(bp_camara, transform_frontal, attach_to=vehiculo)
        camara_frontal.listen(lambda image: procesar_camara_frontal(image, pantalla))
        
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
        pygame.quit()

if __name__ == '__main__':
    main()