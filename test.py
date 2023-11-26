# import cv2

# # Open a connection to the camera (0 by default represents the default camera)
# cap = cv2.VideoCapture(0)

# # Check if the camera is opened successfully
# if not cap.isOpened():
#     print("Error: Could not open camera.")
#     exit()

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     # If the frame is read correctly, ret will be True
#     if ret:
#         # Display the frame
#         cv2.imshow('Camera Feed', frame)

#     # Break the loop if 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the camera and close the window
# cap.release()
# cv2.destroyAllWindows()

import os
import cv2 as cv
import gc
import time
import numpy as np

CONFIDENCE = 0.5
THRESHOLD = 0.4


def getYOLOOutput(img, net):
    blobImg = cv.dnn.blobFromImage(
        img, 1.0/255.0, (416, 416), None, True, False)
    net.setInput(blobImg)
    outInfo = net.getUnconnectedOutLayersNames()
    start = time.time()
    layerOutputs = net.forward(outInfo)
    end = time.time()
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    (H, W) = img.shape[:2]
    boxes = []
    confidences = []
    classIDs = []

    for out in layerOutputs:
        for detection in out:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > CONFIDENCE:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

    dfg = []  # Double-finger-gap
    pc = []  # Palm-center

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            if classIDs[i] == 0:
                dfg.append([x+w/2, y+h/2, confidences[i]])
            else:
                pc.append([x+w/2, y+h/2, confidences[i]])
    return dfg, pc


yolo_dir = 'yolo'
weightsPath = os.path.join(
    yolo_dir, 'yolov4-tiny-obj_best.weights')  # YOLO weights
configPath = os.path.join(yolo_dir, 'yolov4.cfg')  # YOLO config
imgPath1 = 'data/3.jpg'  # Input images
imgPath2 = 'data/4.jpg'


net = cv.dnn.readNetFromDarknet(configPath, weightsPath)
print("[INFO] loading YOLO from disk...")

img1 = cv.imread(imgPath1)
img2 = cv.imread(imgPath2)

dfg1, pc1 = getYOLOOutput(img1, net)
dfg2, pc2 = getYOLOOutput(img2, net)
print(len(dfg1), len(dfg2))
gc.collect()
