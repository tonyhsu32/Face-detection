import time
import cv2

# BGR color space
c_space = {"Dark_Orange": (0, 140, 255), 
         "Orange": (0, 165, 255),
         "Orange_Red": (0, 69, 255)}

class FPS:
    def __init__(self):
        self.begin = time.time()
        self.frames = 0

    def update(self, img, pos = (40, 60), f_scale = 2, f_color = c_space["Dark_Orange"], f_thick = 5):
        end = time.time()
        self.frames += 1

        fps = self.frames / (end - self.begin)
        cv2.putText(img, f"FPS: {int(fps)}", pos, cv2.FONT_HERSHEY_SIMPLEX, f_scale, f_color, f_thick)

        return fps, img


def main():
    fps = FPS()
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("VideoCapture can't receive frame, please try your camera device again ?")
            break

        v_fps, v_img = fps.update(img)
        print(f"fps: {int(v_fps)}")

        cv2.imshow("Camera Frame", v_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Close Stream !")
            break


if __name__ == "__main__":
    main()
