import cv2
import numpy as np
import pickle
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def get_camera_index():
    for i in range(5):
        cam = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cam.isOpened():
            cam.release()
            return i
    return None

def build_squares(img):
    x, y, w, h = 400, 120, 20, 20
    d = 15
    imgCrop = None
    crop = None

    for i in range(10):
        for j in range(5):
            if imgCrop is None:
                imgCrop = img[y:y+h, x:x+w]
            else:
                imgCrop = np.hstack((imgCrop, img[y:y+h, x:x+w]))

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            x += w + d

        if crop is None:
            crop = imgCrop
        else:
            crop = np.vstack((crop, imgCrop))

        imgCrop = None
        x = 400
        y += h + d

    return crop

def get_hand_hist():
    camera_index = get_camera_index()
    if camera_index is None:
        logging.error("❌ No camera detected. Please connect one.")
        return

    cam = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    flagPressedC, flagPressedS = False, False
    imgCrop, hist = None, None

    logging.info("🖐 Press 'c' to calibrate histogram. Press 's' to save and exit.")

    while True:
        ret, img = cam.read()
        if not ret:
            logging.error("❌ Could not read from camera.")
            break

        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and imgCrop is not None:
            hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsvCrop, (0, 20, 70), (20, 255, 255))
            hist = cv2.calcHist([hsvCrop], [0, 1], mask, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            flagPressedC = True
            logging.info("✅ Histogram captured. Press 's' to save or adjust lighting and retry.")

        elif key == ord('s'):
            flagPressedS = True
            break

        imgCrop = build_squares(img)

        if flagPressedC and hist is not None:
            dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            dst = cv2.filter2D(dst, -1, disc)
            blur = cv2.GaussianBlur(dst, (11, 11), 0)
            blur = cv2.medianBlur(blur, 15)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            cv2.imshow("Thresholded Backprojection", cv2.merge([cleaned]*3))

        cv2.imshow("Set Hand Histogram", img)

    cam.release()
    cv2.destroyAllWindows()

    if hist is not None:
        with open("hist", "wb") as f:
            pickle.dump(hist, f)
        logging.info("📦 Histogram saved successfully as 'hist'.")
    else:
        logging.warning("⚠ No histogram captured. Nothing was saved.")

get_hand_hist()
