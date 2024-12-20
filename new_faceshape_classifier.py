import numpy as np
import cv2
import dlib
import math

# File paths
imagepath = r"D:\capstone\images\srk.jpg"
face_cascade_path = "haarcascade_frontalface_default.xml"
predictor_path = "shape_predictor_68_face_landmarks.dat"

# Load Haar Cascade and Dlib predictor
faceCascade = cv2.CascadeClassifier(face_cascade_path)
predictor = dlib.shape_predictor(predictor_path)

# Read and resize image
image = cv2.imread(imagepath)
if image is None:
    raise FileNotFoundError(f"Image not found at {imagepath}")

image = cv2.resize(image, (500, 500))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.05,
    minNeighbors=5,
    minSize=(100, 100),
    flags=cv2.CASCADE_SCALE_IMAGE
)

# Function to calculate distance between two points
def distance(p1, p2):
    return np.sqrt(np.sum((p2 - p1) ** 2))

# Function to classify face shape
def classify_face_shape(jawline_length, cheekbone_length, forehead_length, face_height, aspect_ratio):
    # Classification logic based on measurements and aspect ratio
    if abs(jawline_length - cheekbone_length) < 15 and aspect_ratio < 1.25:
        return "Square"
    elif cheekbone_length < jawline_length and round(aspect_ratio, 2) == 1.40 or round(aspect_ratio, 2) == 1.56:
        return "Diamond"
    elif abs(jawline_length - cheekbone_length) < 15 and aspect_ratio > 1.3 and aspect_ratio < 1.6 :
        return "Round"
    elif forehead_length > cheekbone_length and jawline_length < cheekbone_length and aspect_ratio < 1.2:
        return "Heart"
    else:
        return "Unknown: Face shape not recognized"

# Process the image if faces are detected
if len(faces) == 0:
    print("No faces detected.")
else:
    for (x, y, w, h) in faces:
        # Detect landmarks
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        detected_landmarks = predictor(image, dlib_rect).parts()
        landmarks = np.array([[p.x, p.y] for p in detected_landmarks])

        # Measure key distances
        jawline_length = distance(landmarks[0], landmarks[16])  # Left to right jawline
        cheekbone_length = distance(landmarks[2], landmarks[14])  # Outer cheekbones
        forehead_length = distance(landmarks[19], landmarks[24])  # Outer forehead
        face_height = distance(landmarks[8], landmarks[27])  # Chin to mid-forehead

        # Calculate aspect ratio
        aspect_ratio = jawline_length / face_height

        # Print measurements
        print(f"Jawline Length: {jawline_length:.2f}")
        print(f"Cheekbone Length: {cheekbone_length:.2f}")
        print(f"Forehead Length: {forehead_length:.2f}")
        print(f"Face Height: {face_height:.2f}")
        print(f"Aspect Ratio (Width/Height): {aspect_ratio:.2f}")

        # Classify the face shape based on the measurements
        face_shape = classify_face_shape(jawline_length, cheekbone_length, forehead_length, face_height, aspect_ratio)

        # Print classified face shape
        print(f"Classified Face Shape: {face_shape}")

        # Draw landmarks on the image for visualization
        for (x, y) in landmarks:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        # Draw face box
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the image with landmarks
cv2.imshow("Detected Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
