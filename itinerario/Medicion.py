import json
import cv2
#from yolov8 import YOLOv8Detector  # Importa la clase para la detección YOLOv8
import numpy as np
from Detector_Objetos import *
import serial
import torch
import pandas 
# leemos el modelo
model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'C:\Users\Asus\Documents\itinerario\model\best.pt')
model.conf = 0.5
#marcador aruco 
parametros =  cv2.aruco.DetectorParameters()
diccionario = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

#detector del objeto
detector = DetectorFondoHomogeneo()

# VideoCapture de la cámara
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

# Configura la conexión serial (ajusta el puerto y la velocidad)
arduino_port = 'COM4'  # Cambia esto al puerto COM en Windows
baud_rate = 9600
ser = serial.Serial(arduino_port, baud_rate, timeout=1)
referen = 46
while True:
    # Lectura de la cámara
    ret, frame = cap.read()
    detect = model(frame)

    results = detect.pandas().xyxy[0].to_json(orient="records")
    results = json.loads(results)

    #if not ret:
    #    break
    #deteccion del marcador aruco 
    esquinas, _, _ = cv2.aruco.detectMarkers(frame, diccionario, parameters=parametros)
    esquinas_ent = np.intp(esquinas)
    #cv2.polylines(frame, esquinas_ent, True, (0,0,255),5)
    #perimetro del aruco
    if (len(esquinas_ent)>0):
        perimetro_aruco = cv2.arcLength(esquinas_ent[0], True)
        print(perimetro_aruco)

        #proporcion en cm
        proporcion_cm = perimetro_aruco/ 16
        print(proporcion_cm)

        # Lee una línea desde el puerto serial y convierte a entero
        sensor_reading = ser.readline().decode('utf-8').strip()
        Profundidad = 0    
        try:
            
            sensor_value = int(sensor_reading)
            Profundidad = referen-sensor_value
            print(f"Lectura del sensor: {sensor_value}")
            print(f"Profundidad de caja: {Profundidad}")
        except ValueError:
            print("Lectura no válida")
        #deteccion de objetos
        contornos = detector.deteccion_objetos(frame)
        #print("contornos: ", contornos)
        #dibujo de la deteccion del objeto
        for cont in contornos:
            #cv2.polylines(frame, [cont], True, (0,255,0), 2)
            #rectangulo delimitador
            rectangulo = cv2.minAreaRect(cont)
            (x,y), (an, al), angulo = rectangulo

            #ancho y alto de px a cm
            ancho = an/proporcion_cm
            alto = al/proporcion_cm

            #circulo centro
            cv2.circle(frame,(int(x), int(y)),5, (255,255,0), -1)
            #rectangulo
            rect = cv2.boxPoints(rectangulo) #se obtiene el rectangulo
            rect = np.intp(rect) #se pasa la informacion a entero

            #cv2.polylines(frame, [rect], True, (0,255,0), 2)
            #informacion en pixeles
            
            if en_bbox(x,y, results):
                cv2.putText(frame, "Ancho: {} cm".format(round(ancho,1)), (int(x), int(y-15)), cv2.LINE_AA, 0.8, (150, 0, 255), 2)
                cv2.putText(frame, "Largo: {} cm".format(round(alto,1)), (int(x), int(y+15)), cv2.LINE_AA, 0.8, (75, 0, 75), 2)
                cv2.putText(frame, "Profundidad: {} cm".format(round(Profundidad,1)), (int(x), int(y+50)), cv2.LINE_AA, 0.8, (75, 0, 75), 2)
                print("xy", x, y)
                # Calcular centro del objeto
                objeto_centro_x = int(x)
                objeto_centro_y = int(y)
                
                # Calcular centro del marcador ArUco
                #aruco_centro_x = int(np.mean(esquinas_ent[0][:, 0]))
                #aruco_centro_y = int(np.mean(esquinas_ent[0][:, 1]))
                aruco_centro_x = esquinas_ent[:, :, :, 0].mean()
                aruco_centro_y = esquinas_ent[:, :, :, 1].mean()
                print("ar",aruco_centro_x, aruco_centro_y)
                print(esquinas_ent)
                # Calcular coordenadas relativas
                rel_x = (objeto_centro_x - aruco_centro_x) / proporcion_cm
                rel_y = (objeto_centro_y - aruco_centro_y) / proporcion_cm
                print("rl", rel_x, rel_y)
                # Dibujar fondo blanco detrás de las coordenadas relativas
                rect_bg = np.zeros((70, frame.shape[1], 3), dtype=np.uint8)
                rect_bg[:] = (255, 255, 255)
                
                # Mostrar coordenadas relativas sobre el fondo blanco
                frame[0:70, 0:frame.shape[1]] = rect_bg
                cv2.putText(frame, "Rel X: {} cm".format(round(rel_x, 1)), (20, 40), cv2.LINE_AA, 0.8, (0, 0, 0), 2)
                cv2.putText(frame, "Rel Y: {} cm".format(round(rel_y, 1)), (20, 70), cv2.LINE_AA, 0.8, (0, 0, 0), 2)
                
    # Combinar imagen de detección de objetos y mediciones
    combined_frame = np.vstack((frame, np.squeeze(detect.render())))
    #print("Detec: ", np.squeeze(detect.render()))
    # Mostrar imagen combinada en una sola ventana
    cv2.imshow("Detección de objetos y mediciones", combined_frame)
    

    # Para salir
    t = cv2.waitKey(1)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()