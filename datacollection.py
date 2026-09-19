import cv2
import mediapipe as mp
import numpy as np
import time
import math

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

offset = 20
imgSize = 300
counter = 0

folder = r"C:\Users\mohdy\Downloads\Sign-Language-detection-main\Sign-Language-detection-main\Data\Yes"

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    if result.multi_hand_landmarks:

        handLms = result.multi_hand_landmarks[0]

        # ⭐ Get hand label (Right / Left)
        handedness = result.multi_handedness[0].classification[0].label

        # ⭐ Draw hand skeleton
        mpDraw.draw_landmarks(
            img,
            handLms,
            mpHands.HAND_CONNECTIONS,
            mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

        # ⭐ Compute bounding box
        h, w, c = img.shape
        x_min, y_min = w, h
        x_max, y_max = 0, 0

        for lm in handLms.landmark:
            px, py = int(lm.x * w), int(lm.y * h)
            x_min = min(x_min, px)
            y_min = min(y_min, py)
            x_max = max(x_max, px)
            y_max = max(y_max, py)

        bw, bh = x_max - x_min, y_max - y_min

        # ⭐ Draw bounding box
        cv2.rectangle(img, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (255, 0, 255), 2)

        # ⭐ Draw LEFT/RIGHT text
        cv2.putText(img, handedness, (x_min, y_min - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # ⭐ Prepare crop
        y1 = max(0, y_min - offset)
        y2 = min(img.shape[0], y_max + offset)
        x1 = max(0, x_min - offset)
        x2 = min(img.shape[1], x_max + offset)

        imgCrop = img[y1:y2, x1:x2]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        if imgCrop.size != 0:
            hCrop, wCrop, _ = imgCrop.shape
            aspectRatio = hCrop / wCrop

            if aspectRatio > 1:
                k = imgSize / hCrop
                wCal = math.ceil(k * wCrop)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize

            else:
                k = imgSize / wCrop
                hCal = math.ceil(k * hCrop)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)