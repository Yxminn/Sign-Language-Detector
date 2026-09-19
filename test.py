import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize camera
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# ✔ Correct model path (relative path – works anywhere)
classifier = Classifier(
    r"./Model/keras_model.h5",
    r"./Model/labels.txt"
)

offset = 20
imgSize = 300

# ✔ Labels must match labels.txt
labels = ["Hello", "I love you", "Thank you", "Yes"]

while True:
    success, img = cap.read()
    if not success:
        print("Camera Error: Unable to read frame.")
        break

    imgOutput = img.copy()

    hands, img = detector.findHands(img)  # detect hands

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Prevent negative crop errors
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        if imgCrop.size > 0:
            hCrop, wCrop = imgCrop.shape[:2]
            aspectRatio = hCrop / wCrop

            if aspectRatio > 1:  # tall image
                k = imgSize / hCrop
                wCal = math.ceil(wCrop * k)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize

            else:  # wide image
                k = imgSize / wCrop
                hCal = math.ceil(hCrop * k)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Predict gesture
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print("Prediction:", labels[index])

            # Draw boxes and label
            cv2.rectangle(imgOutput, (x1, y1 - 70), (x1 + 350, y1 - 10), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x1 + 10, y1 - 25),
                        cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (0, 255, 0), 4)

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)