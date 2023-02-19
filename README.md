## Face detection ##

1. **Face_attendance_project.py:**  
Use OpenCV, face_recognition, PIL packages.  
[face_recognition API](https://github.com/ageitgey/face_recognition) 

    - step 1: build class_name and split name with '.jpg'  
    - step 2: transform images to face encoding list  
    - step 3: determin whether the Chinese name  
    - step 4: verify face to create attendance csv  
    - step 5:  
       * use video capture  
       * face locations, face encodings -> compare face and distance  
       * use np.argmin to get min index and print name  
       * get face locations, draw rectangle and use putText to print name  
       
       
2. **Face_detector.py:**  
Use OpenCV, face_recognition packages.  
[face_recognition API](https://github.com/ageitgey/face_recognition) 

    - step 1: find name with '.jpg' and split train and test  
    - step 2: load img -> transform bgr to rgb   
    - step 3: capture face loaction -> face encoding -> draw rectangle  
    - step 4: compare face and distance -> face_compare and face_distance
    
3. **Face_detector_mpSSD.py:**  
Use OpenCV, mediapipe packages.  

    [SSD: Single Shot MultiBox Detector by Google 2016 paper](https://arxiv.org/pdf/1512.02325.pdf)  
    [Mediapipe Python API](https://google.github.io/mediapipe/getting_started/python)

![Stephen Curry](https://github.com/tonyhsu32/Face-detection/blob/main/screenshot/STEPHEN%20CURRY.png)
  
![Guido Van Rossum](https://github.com/tonyhsu32/Face-detection/blob/main/screenshot/GUIDO%20VAN%20ROSSUM.png)

4. **Attend_data, Face_img, screenshot:**  
- Attend_data:  
Attendence person images.
- Face_img:  
Test images.
- screenshot:  
Detected images.

5. **utils.py:**  
FPS tools.

