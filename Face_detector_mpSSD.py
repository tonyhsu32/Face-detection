# import os
import cv2
# import numpy as np
from utils import FPS
import mediapipe as mp

# Using mediapipe(high speed fps):
mp_f_detect = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def mp_face_detection(detect_dis = 0, confidence = 0.7): 
    with mp_f_detect.FaceDetection(model_selection = detect_dis, 
                                   min_detection_confidence = confidence) as f_detector:
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                print("VideoCapture can't receive frame, please try your camera device again ?")
                break

            # resize window
            cv2.namedWindow("Camera Frame", 0)
            cv2.resizeWindow("Camera Frame", 800, 450)

            # improve performance, mark the image is not writeable to pass by reference
            img.flags.writeable = False
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = f_detector.process(img)

            # recover flags.writeable 
            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if res.detections:
                for detect in res.detections:
                    mp_drawing.draw_detection(img, detect)

            # show fps:
            img = cv2.flip(img, 1)

            v_fps, img = fps.update(img)
            print(f"FPS: {v_fps:.2f}")

            # flip image (horithontal) 
            cv2.imshow("Camera Frame", img)

            # quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                print("Stop Stream !")
                cv2.waitKey(0)
            
            if key == ord("q"):
                print("Close Stream !")
                break


if __name__ == "__main__":
    fps = FPS()
    cap = cv2.VideoCapture(0)

    # detect_dis: within 2 meters -> 0, within 5 meters -> 1
    mp_face_detection(detect_dis = 0, confidence = 0.5)

    cap.release()
    cv2.destroyAllWindows()