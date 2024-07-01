import cv2
import numpy as np
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

## Qestion 1

# video_capture = cv2.VideoCapture(0)
# while True:
#     ret, frame = video_capture.read()
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     face = face_classifier.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
#     if len(face) != 0:
#         for (x, y, w, h) in face:
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#     cv2.imshow('Video', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# video_capture.release()
# cv2.destroyAllWindows()

## Question 2

image_dir = 'images/train'
result_dir = 'labels2'
result_files = os.listdir(result_dir)
IoUresults = []
totalIoU = 0
IoUNotNull = 0
totalCount = 0
NotNullCount = 0
CountNotDetected = 0
# For each image
for i in range(0, len(os.listdir(image_dir))):
    filename = os.listdir(image_dir)[i]
    image_path = os.path.join(image_dir, filename)
    image = cv2.imread(image_path) # Open the image

    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert the image to gray
    face = face_classifier.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)) # Detect the face

    result_path = os.path.join(result_dir, filename.replace('.jpg', '.txt').replace('.png', '.txt')) # Get the result file
    result = open(result_path, "r").read().splitlines()
    result = [list(map(str, x.split())) for x in result]

    # For each face detected
    IoU = []
    for (x, y, w, h) in face:
        x1, y1, x2, y2 = x, y, x+w, y+h # Get the coordinates of the face
        currentIoU = 0
        for rect in result:
            _, _, x1_val, y1_val, x2_val, y2_val = rect # Get the coordinates of the result
            x1_val, y1_val, x2_val, y2_val = float(x1_val), float(y1_val), float(x2_val), float(y2_val)
            # Find the top left and bottom right coordinates of the intersection rectangle
            x1_i = max(x1, x1_val)
            y1_i = max(y1, y1_val)
            x2_i = min(x2, x2_val)
            y2_i = min(y2, y2_val)
            # Check if the intersection rectangle is valid
            if x1_i < x2_i and y1_i < y2_i:
                # Calculate the surface of the intersection rectangle
                surface = (x2_i - x1_i) * (y2_i - y1_i)
                # Calculate the surface of the union of the face and the result
                union = (x2 - x1) * (y2 - y1) + (x2_val - x1_val) * (y2_val - y1_val) - surface
                newIoU = surface / union
                #print(newIoU)
                # If the IoU is greater than the current IoU, update the current IoU (we want the best match)
                if newIoU > currentIoU:
                    currentIoU = newIoU
        IoU.append(currentIoU)
        totalIoU += currentIoU
        totalCount += 1
        if currentIoU > 0:
            IoUNotNull += currentIoU
            NotNullCount += 1
    #print(IoU)
    if len(IoU) < len(result):
        for j in range(len(IoU), len(result)):
            IoU.append(0)
            CountNotDetected += 1
    IoUresults.append(IoU)

    if i%100 == 0:
        print("At image", i, ", we have:")
        print("Average IoU (including false positive and undetected faces):", totalIoU/(totalCount+CountNotDetected))
        print("Average IoU not null (only sucessfuly detected faces):", IoUNotNull/NotNullCount)
        print("Precision (Ratio between successful detection and total detections):", NotNullCount/totalCount)
        print("Recall (Ratio between successful detection and total faces):", NotNullCount/(NotNullCount+CountNotDetected))


    # if len(face) != 0:
    #     for i in range (0, len(face)):
    #         (x, y, w, h) = face[i]
    #         if IoU[i] > 0.5:
    #             cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #         else:
    #             cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    # if len(result) != 0:
    #     for (_, _, x, y, w, h) in result:
    #         x, y, w, h = float(x), float(y), float(w), float(h)
    #         cv2.rectangle(image, (int(x), int(y)), (int(w), int(h)), (255, 0, 0), 2)
    # cv2.imshow('Image', image)
    # cv2.waitKey(0)

# cv2.destroyAllWindows()
average_IoU = np.mean(IoUresults)
print("Average IoU:", average_IoU)