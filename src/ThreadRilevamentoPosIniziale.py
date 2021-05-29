import statistics as stat
from threading import Thread

import cv2
import mediapipe as mp
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def distanzaEsecuzione(datiRipInCorso, colonne, dfEsecuzioneCorretta):
    dfRip = pd.DataFrame(datiRipInCorso, columns=colonne)
    valoriDtw = []

    for column in dfRip:
        colonna1 = dfEsecuzioneCorretta[column]
        colonna2 = dfRip[column]

        distanza, path = fastdtw(colonna1, colonna2, dist=euclidean)
        valoriDtw.append(distanza)

    mediana = stat.median(valoriDtw)

    return mediana


def distanzaDaPosIniziale(esercizio, vettorePosIniziale, datiFrameCorrente):
    distanza = 0

    for i in range(66):
        distanza += abs(vettorePosIniziale[i] - datiFrameCorrente[i])  # da fare valore assoluto

    return distanza


class ThreadPosIniziale(Thread):

    def __init__(self, webcam, esercizio):
        Thread.__init__(self)
        self.webcam = webcam
        self.esercizio = esercizio

    def run(self):

        esercizio = self.esercizio

        webcam = self.webcam

        # inizializza modello
        mpPose = mp.solutions.pose
        pose = mpPose.Pose()

        colonne = ['nose_x', 'nose_y',
                   'left_eye_inner_x', 'left_eye_inner_y',
                   'left_eye_x', 'left_eye_y',
                   'left_eye_outer_x', 'left_eye_outer_y',
                   'right_eye_inner_x', 'right_eye_inner_y',
                   'right_eye_x', 'right_eye_y',
                   'right_eye_outer_x', 'right_eye_outer_y',
                   'left_ear_x', 'left_ear_y',
                   'right_ear_x', 'right_ear_y',
                   'mouth_left_x', 'mouth_left_y',
                   'mouth_right_x', 'mouth_right_y',
                   'left_shoulder_x', 'left_shoulder_y',
                   'right_shoulder_x', 'right_shoulder_y',
                   'left_elbow_x', 'left_elbow_y',
                   'right_elbow_x', 'right_elbow_y',
                   'left_wrist_x', 'left_wrist_y',
                   'right_wrist_x', 'right_wrist_y',
                   'left_pinky_x', 'left_pinky_y',
                   'right_pinky_x', 'right_pinky_y',
                   'left_index_x', 'left_index_y',
                   'right_index_x', 'right_index_y',
                   'left_thumb_x', 'left_thumb_y',
                   'right_thumb_x', 'right_thumb_y',
                   'left_hip_x', 'left_hip_y',
                   'right_hip_x', 'right_hip_y',
                   'left_knee_x', 'left_knee_y',
                   'right_knee_x', 'right_knee_y',
                   'left_ankle_x', 'left_ankle_y',
                   'right_ankle_x', 'right_ankle_y',
                   'left_heel_x', 'left_heel_y',
                   'right_heel_x', 'right_heel_y',
                   'left_foot_index_x', 'left_foot_index_y',
                   'right_foot_index_x', 'right_foot_index_y']

        vettorePosIniziale = []

        for i in range(66):
            # carico il dataframe della posizione iniziale
            vettorePosIniziale.append(float(open('../res/' + esercizio + '/DatiPosIniziale.txt', "r").readline()))

        # carico il dataframe dell'esecuzione corretta
        dfEsecuzioneCorretta = pd.read_excel('../res/' + esercizio + '/output.xlsx')

        # flag che diventa true se non viene rilevato nessun landmark utente
        errore = False

        ripetizioneInCorso = False

        eraPosIniziale = False

        almenoUnaRipFatta = False

        # numero di ripetizioni
        ripetizioni = 0

        sogliaPosIniziale = 11  # da sistemare

        sogliaEsecuzione = 25  # DA SISTEMARE

        # inizializza vettore dal quale creare il dataframe
        datiRipInCorso = []

        while True:

            # Svuoto il vettore contenente i landmark del frame corrente
            datiFrameCorrente = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0]

            # legge frame video
            return_value, img = webcam.read()

            width = 500
            height = 300
            dim = (width, height)
            img = cv2.resize(img, dim)

            # converte il frame in RGB
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # rileva i landmarks
            results = pose.process(imgRGB)

            # se trova dei landmarks
            if results.pose_landmarks:

                errore = False

                # per ogni landmark trovato
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    # aggiunge il landmark rilevato al vettore
                    datiFrameCorrente[id * 2] = lm.x
                    datiFrameCorrente[(id * 2) + 1] = lm.y

                # calcolo la distanza tra i dati del frame corrente e i dati della posizione di partenza
                distanzaPosIniziale = distanzaDaPosIniziale(esercizio, vettorePosIniziale, datiFrameCorrente)

                print("distanza pos iniziale {}".format(distanzaPosIniziale))

                eInPosIniziale = distanzaPosIniziale < sogliaPosIniziale

                if eInPosIniziale and not eraPosIniziale:

                    eraPosIniziale = True
                    ripetizioneInCorso = False

                    if almenoUnaRipFatta:
                        valutazioneEsecuzione = distanzaEsecuzione(datiRipInCorso, colonne, dfEsecuzioneCorretta)
                        print("Valutazione esecuzione {}".format(valutazioneEsecuzione))
                        datiRipInCorso = []

                        # se l'esecuzione è corretta
                        if valutazioneEsecuzione < sogliaEsecuzione:
                            ripetizioni += 1
                            print(ripetizioni)

                    else:
                        almenoUnaRipFatta = True

                else:

                    if eraPosIniziale and not eInPosIniziale:
                        ripetizioneInCorso = True
                        eraPosIniziale = False

                if ripetizioneInCorso:
                    datiRipInCorso.append(datiFrameCorrente)

            else:
                if not errore:
                    errore = True
                    print("Posizionati di fronte alla webcam")

# piegamenti: spalla sinistra, spalla destra
# mani
# gomiti
# testa-mani

# squat
# piedi
# ginocchia
# anche-ginocchia
# testa_piedi

# addominali
# piedi-anche
# spalle ginocchia
