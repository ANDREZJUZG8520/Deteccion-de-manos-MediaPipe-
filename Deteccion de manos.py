# Se importa la biblioteca de open cv y el framework de mediapipe

import cv2
import mediapipe as mp


# llamamos la funcion solutions para que dibuje las lineas y especificamos que sean las de las manos

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


# Aqui llamamos a la camara del computador para que se active 

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


# En esta parte le damos las configuraciones que debe tener el lector de de las manos

with mp_hands.Hands(
	# Esta funcion permite que el programa detecte los puntos de la mano cada que se modifique la posicion de esta
    static_image_mode=False,
    #Esta funcion se especifica cuantas manos quiere que identifique
    max_num_hands=2,
    # Esta funcion es la que refresca los detectores para que sea en impo real 
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

        # Se le colocan valores de RGB a la imagen ya que open cv trabaja en BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Direccionamos la imagen procesada del stream al hand.process
        results = hands.process(frame_rgb)

        # En esta parte realizamos la lectura de las manos y creamos las lineas y los puntos de ellas

        if results.multi_hand_landmarks is not None:

        	#Se realiza el ciclo for para detectar los 21 puntos por cada mano detectada
            for hand_landmarks in results.multi_hand_landmarks:

            	#Aca se dibujan las lineas y puntos de las manos llamando a la libreria mediapipe paa su creacion
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,

                    # Se crean los puntos y lineas especificando su tamaño y color
                    mp_drawing.DrawingSpec(color=(50,0,200), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255,0,255), thickness=3, circle_radius=5))

        # Mostramos los frames en pantalla con el open cv
        cv2.imshow('Frame',frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Le damos el funcionamiento al programa de la camara y del open cv para para que ejecute correctamente
cap.release()
cv2.destroyAllWindows()

