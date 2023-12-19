import os

import cv2

model_file = "./file/res10_300x300_ssd_iter_140000_fp16.caffemodel"
config_file = "./file/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_file, model_file)
threshold = 0.9


def Catch(input_img):
    print(input_img.shape)

    frameHeight = input_img.shape[0]
    frameWidth = input_img.shape[1]

    blob = cv2.dnn.blobFromImage(input_img, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        detection_score = detections[0, 0, i, 2]
        if detection_score > threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            cv2.rectangle(input_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    file_path = "./Data/Finish"
    cv2.imwrite(os.path.join(file_path, 'found_face.jpg'), input_img)


