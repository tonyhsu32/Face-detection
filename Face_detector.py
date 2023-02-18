import os
import re
import cv2
import face_recognition
import numpy as np

dir_path = os.getcwd() + "/Face_img/"

# step 1: find name with '.jpg' and split train and test
def split_name(path):
    class_name = []
    for img in np.array(os.listdir(path)):
        if re.search(r'\stest\.jpg$', img):
            if img != []:
                img_name = os.path.splitext(img)[0]
                class_name.append(img_name[:-5])
    return class_name

class_name = split_name(dir_path)
print(class_name)

img_name = class_name[6] # A few Imgs Exception -> face_locations: not detected


# step 2: load img -> transform bgr to rgb 
img = face_recognition.load_image_file(dir_path+img_name+".jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_test = face_recognition.load_image_file(dir_path+img_name+" test.jpg")
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)


# step 3: capture face loaction -> face encoding -> draw rectangle
# face_loc -> (y1, x2, y2, x1)
face_loc = face_recognition.face_locations(img)[0]
face_encode = face_recognition.face_encodings(img)[0]
cv2.rectangle(img, (face_loc[3], face_loc[0]), (face_loc[1], face_loc[2]), (0, 0, 255), 2)

face_loc_test = face_recognition.face_locations(img_test)[0]
face_encode_test = face_recognition.face_encodings(img_test)[0]
cv2.rectangle(img_test, (face_loc_test[3], face_loc_test[0]), (face_loc_test[1], face_loc_test[2]), (0, 0, 255), 2)


# step 4: compare face and distance -> face_compare and face_distance
compare_res = face_recognition.compare_faces([face_encode], face_encode_test)
face_dis = face_recognition.face_distance([face_encode], face_encode_test)
print(f"Is the same person in two imgs? {compare_res}")
print(f"Face distance: {face_dis}")

cv2.putText(img_test, f"Face: {compare_res} Distance: {round(face_dis[0], 2)}", (50, 50), 
            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)


cv2.imshow(img_name, img)
cv2.waitKey(0)
cv2.imshow(img_name+" test", img_test)
cv2.waitKey(0)




