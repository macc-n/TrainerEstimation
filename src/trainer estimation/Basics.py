import cv2
import mediapipe as mp


class Training:
aa
    def __init__(self, esercizio):
        self.esercizio = esercizio

    def Training(self):
        esercizio = self.esercizio

        # inizializza modello
        mpPose = mp.solutions.pose
        pose = mpPose.Pose()

        cap = cv2.VideoCapture('../../res/' + esercizio + '/video.mp4')

        while True:
            # legge frame video
            success, img = cap.read()
            img = cv2.flip(img, 1)

            # converte il frame in RGB
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # rileva i landmarks
            results = pose.process(imgRGB)
            return results
