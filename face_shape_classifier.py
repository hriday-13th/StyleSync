import numpy as np
import cv2
import dlib
from math import degrees

face_cascade_path = "/home/hprad/Projects/capstone/haarcascade_frontalface_default.xml"
predictor_path = "/home/hprad/Projects/capstone/shape_predictor_68_face_landmarks.dat"

def classify_face_shape(img):
    imagepath = img
    faceCascade = cv2.CascadeClassifier(face_cascade_path)
    predictor = dlib.shape_predictor(predictor_path)

    def distance(p1, p2):
        return np.sqrt(np.sum((p2 - p1) ** 2))

    def calculate_angle(pointA, pointB, pointC):
        a = np.array(pointA)
        b = np.array(pointB)
        c = np.array(pointC)

        ab = b - a
        cb = b - c

        cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    
    def classify_face_shape(jawline_length, cheekbone_length, forehead_length, aspect_ratio):
        if abs(jawline_length - cheekbone_length) < 15 and aspect_ratio < 1.25:
            return "Square"
        elif cheekbone_length < jawline_length and round(aspect_ratio, 2) == 1.40 or round(aspect_ratio, 2) == 1.56:
            return "Diamond"
        elif abs(jawline_length - cheekbone_length) < 15 and aspect_ratio > 1.3 and aspect_ratio < 1.6 :
            return "Round"
        elif forehead_length > cheekbone_length and jawline_length < cheekbone_length and aspect_ratio < 1.2:
            return "Oblong"
        else:
            return "Unknown: Face shape not recognized"


    image = cv2.imread(imagepath)
    if image is None:
        raise FileNotFoundError(f"Image not found at {imagepath}")

    image = cv2.resize(image, (500, 500))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gray, (3, 3), 0)

    faces = faceCascade.detectMultiScale(
        gauss,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        detected_landmarks = predictor(image, dlib_rect).parts()
        landmarks = np.array([[p.x, p.y] for p in detected_landmarks])

        jawline_length = distance(landmarks[0], landmarks[16])
        cheekbone_length = distance(landmarks[2], landmarks[14])
        forehead_length = distance(landmarks[19], landmarks[24])
        face_height = distance(landmarks[8], landmarks[27])

        aspect_ratio = jawline_length / face_height
        face_shape = classify_face_shape(jawline_length, cheekbone_length, forehead_length, aspect_ratio)

    return face_shape