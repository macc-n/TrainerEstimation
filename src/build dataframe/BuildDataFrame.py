import cv2
import mediapipe as mp
import pandas as pd


class DataFrameBuilder:

    def buildDataframe(self, path):

        # inizializza il dataframe
        df = pd.DataFrame()

        # inizializza modello
        mpPose = mp.solutions.pose
        pose = mpPose.Pose()

        # carica video
        cap = cv2.VideoCapture(path)

        # inizializza vettore che conterrà le coordinate dei landmarks rilevati in un frame
        data1 = []

        try:
            while True:aa

                data = []

                # legge frame video
                success, img = cap.read()
                img = cv2.flip(img, 1)

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

                    # per ogni landmark trovato
                    for id, lm in enumerate(results.pose_landmarks.landmark):
                        # aggiunge il landmark rilevato ad un vettore temporaneo
                        data.append(lm.x)
                        data.append(lm.y)
                    # aggiunge il vettore temporaneo al vettore principale
                    data1.append(data)
        except:
            df = pd.DataFrame(data1, columns=['nose_x', 'nose_y',
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
                                              'right_foot_index_x', 'right_foot_index_y'])
        return df


def main():
    bdf = DataFrameBuilder()
    df = bdf.buildDataframe('../../res/flessioni/video.mp4')
    df.to_excel('../../res/flessioni/dataframe.xlsx')


if __name__ == "__main__":
    main()
