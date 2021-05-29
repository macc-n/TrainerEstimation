import time

import cv2
import mediapipe as mp

import Basics
from src.ThreadRilevamentoPosIniziale import ThreadPosIniziale


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
        thread = ThreadPosIniziale(webcam, esercizio)
        thread.start()

        # mostra il video all'utente
        while True:

            # legge frame video
            success, img = webcam.read()

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

            # calcola gli fps
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            # scrive gli fps sull'immagine
            cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            # mostra il frame
            cv2.imshow('img', img)
            cv2.waitKey(1)


class main:

    def __init__(self, esercizio):
        self.esercizio = esercizio

    def main(self):
        detector = Interfaccia(self.esercizio)
        print("Mettiti in posizione, aiutati con le linee guida")
        detector.Interfaccia()
