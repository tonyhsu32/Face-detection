import os
import re
from datetime import datetime
import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils import FPS

dir_path = os.getcwd() + "/Attend_data"

# step 1: build class_name and split name with '.jpg'
def build_and_split(path):
    images = []
    class_name = []
    for img in np.array(os.listdir(path)):
        if re.search(r'.*\.jpg$', img):
            if img != []:
                images.append(img)
                class_name.append(os.path.splitext(img)[0])
    return images, class_name

images, class_name = build_and_split(dir_path)
print(f"Face_images: {images}")
print(f"Face_names: {class_name}")


# step 2: transform images to face encoding list
def encoding_images(path, imgs):
    encode_list = []
    for img in imgs:
        img = cv2.imread(f"{path}/{img}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_encode = face_recognition.face_encodings(img_rgb)[0]
        encode_list.append(img_encode)
    return encode_list

encode_list = encoding_images(dir_path, images)
# print(encode_list)


# step 3: determin whether the Chinese name: 
# putText() does not support Chinese -> transform ??? to Chinese
def trans_to_chinese(image, name, loc):
    x1, x2, y2 = loc
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    # fonts: NotoSansCJK-Bold.ttc or SimHei.ttf
    font = ImageFont.truetype(os.getcwd()+"/Fonts/NotoSansCJK-Bold.ttc", 35, encoding = "utf-8")
    # x: x1+(x2-x1)//3 or x1
    draw.text((x1, y2-42), name, (255, 255, 255), font = font)  
    pil_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return pil_img


# step 4: verify face to create attendance csv
def export_attendance_csv(name):
    with open(os.getcwd()+"/Attendance_book.csv", "r+") as f:
        orig_list = f.readlines()
        data_list = []
        for data in orig_list:
            sub_data = data.split(',')
            data_list.append(sub_data[0])
        if name not in data_list:
            t = datetime.now()
            f.writelines(f"\n{name}, {t.strftime('%H:%M:%S')}")

# export_attendance_csv('Stephen Curry')


# step 5: 
# 1. use video capture 
# 2. face locations, face encodings -> compare face and distance
# 3. use np.argmin to get min index and print name
# 4. get face locations, draw rectangle and use putText to print name   


# use class is slow -> FPS ≈ 0.5x real-time
# detect algorithm: dlib's HOG + Linear SVM
# recognize algorithm: CNN(ResNet)
class face_recognition_video():
    def __init__(self):
        self.fps = FPS()
        self.cap = cv2.VideoCapture(0)
        self.face_api = face_recognition

    # step 2: transform images to face encoding list
    def encoding_images(self, path, imgs):
        encode_list = []
        for img in imgs:
            img = cv2.imread(f"{path}/{img}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_encode = self.face_api.face_encodings(img_rgb)[0]
            encode_list.append(img_encode)
        return encode_list

    def process_image(self, img):
        # img_resized = cv2.resize(v_img, (0, 0), None, 0.25, 0.25)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faceLoc = self.face_api.face_locations(img)
        faceEncode = self.face_api.face_encodings(img, faceLoc)

        return faceLoc, faceEncode, img
    
    def release_cam(self):
        self.cap.release()
        cv2.destroyAllWindows()
        
    def process_video(self, v_size = (720, 400)):
        # resize window
        cv2.namedWindow("Camera Frame", 0)
        cv2.resizeWindow("Camera Frame", v_size[0], v_size[1])

        images, class_name = build_and_split(dir_path)
        encode_list = self.encoding_images(dir_path, images)

        while self.cap.isOpened():
            res, v_img = self.cap.read()

            if not res:
                print("VideoCapture can't receive frame, please try your camera device again ?")
                break
            
            faceLoc, faceEncode, processed_img = self.process_image(v_img)
            
            for face_loc, face_encode in zip(faceLoc, faceEncode):
                # compare_res = self.face_api.compare_faces(encode_list, face_encode)
                face_dis = self.face_api.face_distance(encode_list, face_encode)
        
                # get name with min dis index
                min_index = np.argmin(face_dis)
                name = class_name[min_index].upper()
                print(name)

                # face_locations => (y1, x2, y2, x1)
                # y1, x2, y2, x1 = tuple(i*4 for i in face_loc) 
                y1, x2, y2, x1 = face_loc
        
                # not use resize img (if camera img size is small)
                cv2.rectangle(processed_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.rectangle(processed_img, (x1, y2-35), (x2, y2), (0, 200, 0), cv2.FILLED)
                
                # transform name to Chinese or English
                pattern = re.compile(u"[\u4e00-\u9fa5]+")
                if pattern.fullmatch(name):
                    trans_img = trans_to_chinese(processed_img, name, (x1, x2, y2))
                else:
                    cv2.putText(processed_img, name, (x1-40, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    trans_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
                
                # used resize img (if camera img size is big)
                # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # cv2.rectangle(img, (x1, y2-6), (x2, y2), (0, 200, 0), cv2.FILLED)
                # cv2.putText(img, name, (x1-25, y2+3), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 2)
                
                # plot fps
                v_fps, export_img = self.fps.update(trans_img)
                print(f"FPS: {v_fps:.2f}")
                
                # export attendance csv
                export_attendance_csv(name)
                cv2.imshow("Camera Frame", export_img)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    print("Stop Stream !")
                    cv2.waitKey(0)

                if key == ord('q'):
                    print("Close Stream !")
                    break

        self.release_cam()
                

if __name__ == "__main__":
    # excute face recognition
    face_object = face_recognition_video()
    face_object.process_video()