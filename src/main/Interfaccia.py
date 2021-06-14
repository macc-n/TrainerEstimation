import cv2
import keyboard
import mediapipe as mp

import Basics
from src.main.ThreadRilevamento import ThreadEsecuzione


class Interfaccia:

    def __init__(self, esercizio):
        self.esercizio = esercizio

    def Interfaccia(self):

        esercizio = self.esercizio

        # inizializza webcam
        webcam = cv2.VideoCapture(0)

        # inizializza modello
        mpPose = mp.solutions.pose
        pose = mpPose.Pose()

        # per disegnare i landmarks
        mpDraw = mp.solutions.drawing_utils

        pTime = 0

        # ricava i landmarks per la posizione di partenza
        b = Basics.Training(esercizio)
        resultsTraining = b.Training()

        # verifica quando l'utente si mette in posizione di partenza
        thread = ThreadEsecuzione(webcam, esercizio)
        thread.start()

        # mostra il video all'utente
        while True:

            # legge frame video
            success, img = webcam.read()
            img = cv2.flip(img, 1)

            # converte il frame in RGB
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # rileva i landmarks
            results = pose.process(imgRGB)

            # disegna i landmarks per la posizione di partenza
            mpDraw.draw_landmarks(img, resultsTraining.pose_landmarks, mpPose.POSE_CONNECTIONS)

            # se trova dei landmarks
            if results.pose_landmarks:
                # disegna i landmarks e le connessioni sull'immagine
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

            # mostra il frame
            cv2.imshow('img', img)
            cv2.waitKey(1)
            if keyboard.is_pressed('q'):
                break


class main:

    def __init__(self, esercizio):
        self.esercizio = esercizio

    def main(self):
        detector = Interfaccia(self.esercizio)
        print("Mettiti in posizione, aiutati con le linee guida")
        detector.Interfaccia()
