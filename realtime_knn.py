import cv2
import numpy as np
import joblib
from skimage.feature import hog

def extract_hog_features(img):
    img = cv2.resize(img, (80, 80))
    features, _ = hog(img, orientations=9, pixels_per_cell=(8,8),
                      cells_per_block=(2,2), visualize=True, block_norm='L2-Hys')
    return features

haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
knn, scaler, id_map = joblib.load(r"D:/Face_Dection/face_knn2.pkl")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img = cap.read()
    if not success: break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        feat = extract_hog_features(roi).reshape(1, -1)
        feat = scaler.transform(feat)
        pred = knn.predict(feat)[0]
        name = id_map[pred]

        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Face Detection (KNN + HOG)", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
