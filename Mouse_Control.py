#Llamamos a las Librerias y frameworks que utilizaremos

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# llamamos la funcion solutions para que dibuje las lineas y especificamos que sean las de las manos
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Aqui llamamos a la camara del computador para que se active
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Le damos el color al puntero del mouse
color_mouse = (255, 0, 255)

# Seleccionamos la parte de la pantalla del computador que usaremos
SCREEN_X_INI = 100
SCREEN_Y_INI = 100
SCREEN_X_FIN = 900
SCREEN_Y_FIN = 1200

# Aca realizamos la operacion para calcular el tamaño del rectangulo para el monitor y el rectangulo del stream
aspect_ratio_screen = (SCREEN_X_FIN - SCREEN_X_INI) / (SCREEN_Y_FIN - SCREEN_Y_INI)
print("aspect_ratio_screen:", aspect_ratio_screen)

#Colocamos el punto de inicio del recuadro en el stream para manejar con la mano
X_Y_INI = 100


# Se crea esta funcion para calcular las distancias de los puntos de los dedos
def distancia_dedos(x1, y1, x2, y2):
    # Se colocan los 2 puntos para hallar la distancia entre los 2 puntos y restornarla
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    return np.linalg.norm(p1 - p2)

# Se crea la funcion para detectar cuando se baje el dedo 
def detector_click(hand_landmarks):
    finger_down = False

    # Se les coloca el color a los puntos
    color_base = (255, 0, 112)
    color_indice = (255, 198, 82)

    # En esta parte se uiliza la teoria de los  27 puntos de mediapipe y se organiza para que el punto del dedo
    #al tocar el punto de la palma de la mano (0) lo identifique

    # Este es el punto de la palma de la mano
    x_base1 = int(hand_landmarks.landmark[0].x * width)
    y_base1 = int(hand_landmarks.landmark[0].y * height)

    # Este es el punto de la base del dedo medio, donde esta el puntero
    x_base2 = int(hand_landmarks.landmark[9].x * width)
    y_base2 = int(hand_landmarks.landmark[9].y * height)

    # Este punto es el de la punta del dedo indice
    x_index = int(hand_landmarks.landmark[8].x * width)
    y_index = int(hand_landmarks.landmark[8].y * height)


    # Se llama a la funcion de calcular la distacia de los puntos para calcular entre la palma y la base del dedo medio
    d_base = distancia_dedos(x_base1, y_base1, x_base2, y_base2)

    # Esta es la distancia entre la base y la punta del dedo indice
    d_base_index = distancia_dedos(x_base1, y_base1, x_index, y_index)

    # Se crea una condicion comprobar si la las distancias de los puntos indican que se cerro la mano, si es asi la variable finger_down es verdadera
    if d_base_index < d_base:
        finger_down = True
        # Se le varian los colores al cerrar la mano
        color_base = (255, 0, 255)
        color_indice = (255, 0, 255)

    # Aca se dibujan los 2 puntos y las lineas que conformarian la mano, ya que el puntero ya tiene circulo
    cv2.circle(output, (x_base1, y_base1), 5, color_base, 2)
    cv2.circle(output, (x_index, y_index), 5, color_indice, 2)
    cv2.line(output, (x_base1, y_base1), (x_base2, y_base2), color_base, 3)
    cv2.line(output, (x_base1, y_base1), (x_index, y_index), color_indice, 3)

    # Se retorna la variable de la funcion
    return finger_down


# En esta parte le damos las configuraciones que debe tener el lector de de la mano que utilizaremos
with mp_hands.Hands(
    # Esta funcion permite que el programa detecte los puntos de la mano cada que se modifique la posicion de esta
    static_image_mode=False,
    # Esta funcion se especifica cuantas manos va a identificar y en este caso solo sera 1
    max_num_hands=1,
    # Esta funcion es la que refresca los detectores para que sea en tiempo real 
    min_detection_confidence=0.5) as hands:


# En esta parte se hace la lectura de la camara por frames
    while True:

        # Se capta cada frame del stream
        ret, frame = cap.read()
        if ret == False:
            break

        # Se toma el ancho y el alto de cada fotograma
        height, width, _ = frame.shape
        # Se voltea la imagen del stream para ver al derecho el video
        frame = cv2.flip(frame, 1)

        # Aca se hace l operacion para que el recuadro del stream sea igual de proporcional al del computador
        area_width = width - X_Y_INI * 2
        # Operamos y convertimos el area del computador al del stream por el aspecto del radio calculado antes
        area_height = int(area_width / aspect_ratio_screen)
        # Aca creamos el recuadro en el cual se va a mover la mano
        aux_image = np.zeros(frame.shape, np.uint8)
        # Se le coloca el tamaño calculado antes y se le da color
        aux_image = cv2.rectangle(aux_image, (X_Y_INI, X_Y_INI), (X_Y_INI + area_width, X_Y_INI +area_height), (0, 0, 255), -1)

        # Aca se combinan el video stream y la imagen del rectangulo, dandole transparencia de 0.8
        output = cv2.addWeighted(frame, 1, aux_image, 0.8, 0)

        # Se le colocan valores de RGB a la imagen ya que open cv trabaja en BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Direccionamos la imagen procesada del stream al hand.process
        results = hands.process(frame_rgb)
        

        # En esta parte realizamos la lectura de las manos y creamos las lineas y los puntos de ellas
        if results.multi_hand_landmarks is not None:

            #Se realiza el ciclo for para detectar el punto que se va a detectar
            for hand_landmarks in results.multi_hand_landmarks:

                #Aca se llama a la funcion que identifica el punto 9 de la mano, que segun mediapipe es la base del dedo medio
                x = int(hand_landmarks.landmark[9].x * width)
                y = int(hand_landmarks.landmark[9].y * height)

                # Realizamos la operacion para que el puntero del mouse y el de la mano se proyecten correctamente en las mismas proporciones
                xm = np.interp(x, (X_Y_INI, X_Y_INI + area_width), (SCREEN_X_INI, SCREEN_X_FIN))
                ym = np.interp(y, (X_Y_INI, X_Y_INI + area_height), (SCREEN_Y_INI, SCREEN_Y_FIN))

                # LLamamos la libreria pyautogui para que detecte el mouse de computador y le damos los valores del puntero de la mano
                pyautogui.moveTo(int(xm), int(ym))

                # Se llama a la funcion del dedo para que la libreria pyautogui le de click al cerrar la mano
                if detector_click(hand_landmarks):
                    pyautogui.click()

                # Se crea y configura el puntero que se detecta en la mano
                cv2.circle(output, (x, y), 10, color_mouse, 3)
                cv2.circle(output, (x, y), 5, color_mouse, -1)

        #Se visualiza la imagen del stream con el rectangulo encima llamada output
        cv2.imshow('output', output)
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Le damos el funcionamiento al programa de la camara y del open cv para para que ejecute correctamente
cap.release()
cv2.destroyAllWindows()