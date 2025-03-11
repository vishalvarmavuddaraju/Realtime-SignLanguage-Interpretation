import cv2
import numpy as np
import pickle

def get_camera_index():
    """Finds a working camera index to avoid 'Camera index out of range' error."""
    for i in range(5):  # Try indexes 0 to 4
        cam = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow for Windows
        if cam.isOpened():
            cam.release()
            return i
    return None  # No cameras found

def get_hand_hist():
    camera_index = get_camera_index()
    if camera_index is None:
        print("❌ Error: No camera detected!")
        return

    cam = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Ensure stable capture

    flagPressedC, flagPressedS = False, False
    imgCrop = None
    hist = None  # Ensure hist is initialized

    while True:
        ret, img = cam.read()
        if not ret:
            print("❌ Error: Could not read frame from camera.")
            break

        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord('c') and imgCrop is not None:
            hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
            flagPressedC = True
            hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

        elif keypress == ord('s'):
            flagPressedS = True
            break

        if flagPressedC and hist is not None:
            dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            cv2.filter2D(dst, -1, disc, dst)
            blur = cv2.GaussianBlur(dst, (11, 11), 0)
            blur = cv2.medianBlur(blur, 15)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = cv2.merge((thresh, thresh, thresh))
            cv2.imshow("Thresh", thresh)

        if not flagPressedS:
            imgCrop = build_squares(img)  # ✅ Bigger squares

        cv2.imshow("Set Hand Histogram", img)

    cam.release()
    cv2.destroyAllWindows()

    if hist is not None:
        with open("hist", "wb") as f:  # ✅ Save as 'hist' without '.pkl'
            pickle.dump(hist, f)
        print("✅ Histogram saved successfully as 'hist'.")
    else:
        print("⚠ Warning: No histogram was captured, so nothing was saved.")

def build_squares(img):
    x, y, w, h = 400, 120, 20, 20  # ✅ Bigger squares
    d = 15  # ✅ Increased spacing
    imgCrop = None
    crop = None

    for i in range(10):  # 10 rows
        for j in range(5):  # 5 columns
            if imgCrop is None:
                imgCrop = img[y:y+h, x:x+w]
            else:
                imgCrop = np.hstack((imgCrop, img[y:y+h, x:x+w]))

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # ✅ Thicker border
            x += w + d  # Move to the right

        if crop is None:
            crop = imgCrop
        else:
            crop = np.vstack((crop, imgCrop))

        imgCrop = None
        x = 400  # Reset x
        y += h + d  # Move down

    return crop

get_hand_hist()
