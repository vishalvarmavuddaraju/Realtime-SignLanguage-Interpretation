import cv2
import numpy as np
import pickle, os, sqlite3, random, logging

image_x, image_y = 50, 50
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def get_hand_hist():
    with open("hist", "rb") as f:
        hist = pickle.load(f)
    return hist

def get_camera_index():
    for i in range(5):
        cam = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cam.isOpened():
            cam.release()
            return i
    return None

def init_create_folder_database():
    if not os.path.exists("gestures"):
        os.mkdir("gestures")
    if not os.path.exists("gesture_db.db"):
        conn = sqlite3.connect("gesture_db.db")
        create_table_cmd = "CREATE TABLE gesture ( g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, g_name TEXT NOT NULL )"
        conn.execute(create_table_cmd)
        conn.commit()

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

def store_in_db(g_id, g_name):
    conn = sqlite3.connect("gesture_db.db")
    cmd = "INSERT INTO gesture (g_id, g_name) VALUES (?, ?)"
    try:
        conn.execute(cmd, (g_id, g_name))
    except sqlite3.IntegrityError:
        choice = input("g_id already exists. Want to change the record? (y/n): ")
        if choice.lower() == 'y':
            cmd = "UPDATE gesture SET g_name = ? WHERE g_id = ?"
            conn.execute(cmd, (g_name, g_id))
            conn.commit()
            print(f"✅ Gesture {g_id} updated in DB.")

            # Delete files 1201 to 2400 in the gesture folder
            folder_path = f"gestures/{g_id}"
            deleted = 0
            for i in range(1201, 2401):
                file_path = os.path.join(folder_path, f"{i}.jpg")
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted += 1
            print(f"🗑️ Deleted {deleted} files from {folder_path} (1201–2400).")
        else:
            print("Doing nothing...")
            return
    else:
        conn.commit()
    conn.close()

def store_images(g_id):
    total_pics = 1200

    camera_index = get_camera_index()
    if camera_index is None:
        logging.error("❌ No camera detected. Please connect one.")
        return

    cam = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    x, y, w, h = 270, 100, 350, 350

    create_folder("gestures/" + str(g_id))
    pic_no = 0
    flag_start_capturing = False
    frames = 0
    hist = None

    logging.info("🖐 Press 'c' to calibrate histogram. Press 's' to start capturing.")

    while True:
        ret, img = cam.read()
        if not ret:
            logging.error("❌ Could not read frame from camera.")
            break

        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        crop = img[y:y+h, x:x+w]
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_crop, (0, 20, 70), (20, 255, 255))
            hist = cv2.calcHist([hsv_crop], [0, 1], mask, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            logging.info("✅ Histogram captured. Press 's' to start capturing.")

        elif key == ord('s'):
            if hist is None:
                logging.warning("⚠ Histogram not captured yet. Press 'c' to calibrate first.")
            else:
                logging.info("📸 Starting to capture images...")
                break

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "Press 'c' to calibrate, 's' to start capture", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Capturing gesture", img)

    while pic_no < total_pics:
        ret, img = cam.read()
        if not ret:
            logging.error("❌ Could not read frame from camera.")
            break

        img = cv2.flip(img, 1)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)

        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        dst = cv2.filter2D(dst, -1, disc)
        blur = cv2.GaussianBlur(dst, (11, 11), 0)
        blur = cv2.medianBlur(blur, 15)

        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh = thresh[y:y+h, x:x+w]

        cv2.imshow("Thresholded Backprojection", thresh)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 10000 and frames > 5:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                pic_no += 1
                save_img = thresh[y1:y1+h1, x1:x1+w1]

                if w1 > h1:
                    save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2), int((w1-h1)/2), 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                elif h1 > w1:
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2), int((h1-w1)/2), cv2.BORDER_CONSTANT, (0, 0, 0))

                save_img = cv2.resize(save_img, (image_x, image_y))
                if random.randint(0, 10) % 2 == 0:
                    save_img = cv2.flip(save_img, 1)

                cv2.putText(img, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))
                cv2.imwrite("gestures/" + str(g_id) + "/" + str(pic_no) + ".jpg", save_img)

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        cv2.imshow("Capturing gesture", img)

        keypress = cv2.waitKey(1)
        if keypress == ord('c'):
            flag_start_capturing = not flag_start_capturing
            frames = 0 if not flag_start_capturing else frames

        if flag_start_capturing:
            frames += 1

    cam.release()
    cv2.destroyAllWindows()

# ========== Main ==========
init_create_folder_database()
g_id = input("Enter gesture no.: ")
g_name = input("Enter gesture name/text: ")
store_in_db(g_id, g_name)
store_images(g_id)
