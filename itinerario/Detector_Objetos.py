import cv2

class DetectorFondoHomogeneo():
    def __init__(self):
        pass

    def deteccion_objetos(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)
        #ENCUENTRA CONTORNOS
        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objetos_contornos =  []

        #si encontramos contornos entramos al for
        for cnt in contornos:
            area = cv2.contourArea(cnt)
            #si el area es mayor a 2000 agregamos el objeto a la lista
            if area > 2000:
                objetos_contornos.append(cnt)

        return objetos_contornos

def en_bbox(x, y, results):
    for bbox in results:
        xmin, ymin, xmax, ymax = bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
        if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
            return True
    return False     