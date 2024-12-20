import cv2
import numpy as np
import mediapipe as mp
import urllib.request
from mediapipe import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions


url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
output_path = "pose_landmarker.task"
urllib.request.urlretrieve(url, output_path)

model = "/content/pose_landmarker.task"


def classify_body_shape(image_path, mp_image, gender):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img is None:
        return "Image not found. Please check the path."

    base_options = BaseOptions(model_asset_path='pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)

    image = mp_image
    
    detection_result = detector.detect(image)

    landmark_details = {}

    landmark_labels = {
        12: 'Left shoulder',
        13: 'Right shoulder',
        24: 'Left hip',
        25: 'Right hip'
    }

    for i, landmarks_list in enumerate(detection_result.pose_landmarks):
        for j, landmark in enumerate(landmarks_list):
            if j + 1 in landmark_labels:
                label = landmark_labels[j + 1]
                landmark_list = [landmark.x, landmark.y]
                landmark_details[label] = landmark_list

    edges = cv2.Canny(img,100,200)

    def calculate_midpoint(point1, point2, ratio):
        x = point1[0] + (point2[0] - point1[0]) * ratio
        y = point1[1] + (point2[1] - point1[1]) * ratio
        return (x, y)

    def create_division_points(landmark_details):
        left_shoulder = landmark_details['Left shoulder']
        right_shoulder = landmark_details['Right shoulder']
        left_hip = landmark_details['Left hip']
        right_hip = landmark_details['Right hip']

        mid1 = calculate_midpoint(left_shoulder, right_shoulder, 0.5)
        mid2 = calculate_midpoint(left_hip, right_hip, 0.5)

        point1 = calculate_midpoint(left_shoulder, left_hip, 1/3)
        point2 = calculate_midpoint(right_shoulder, right_hip, 1/3)
        point3 = calculate_midpoint(mid1, mid2, 1/2)
        point4 = point3
        point5 = calculate_midpoint(mid1, mid2, 2.4/3)
        point6 = point5

        division_points = {
            'left_shoulder': left_shoulder,
            'right_shoulder': right_shoulder,
            'left_bust': point1,
            'right_bust': point2,
            'left_waist': point3,
            'right_waist': point4,
            'left_high_hip': point5,
            'right_high_hip': point6,
            'left_hip': left_hip,
            'right_hip': right_hip
        }
        return division_points

    division_points = create_division_points(landmark_details)

    width, height = img.shape[1], img.shape[0]

    left_shoulder_pixel = (int(division_points['left_shoulder'][0] * width), int(division_points['left_shoulder'][1] * height))
    right_shoulder_pixel = (int(division_points['right_shoulder'][0] * width), int(division_points['right_shoulder'][1] * height))

    left_bust_pixel = (int(division_points['left_bust'][0] * width), int(division_points['left_bust'][1] * height))
    right_bust_pixel = (int(division_points['right_bust'][0] * width), int(division_points['right_bust'][1] * height))

    left_waist_pixel = [int(division_points['left_waist'][0] * width), int(division_points['left_waist'][1] * height)]
    right_waist_pixel = [int(division_points['right_waist'][0] * width), int(division_points['right_waist'][1] * height)]

    left_high_hip_pixel = (int(division_points['left_high_hip'][0] * width), int(division_points['left_high_hip'][1] * height))
    right_high_hip_pixel = (int(division_points['right_high_hip'][0] * width), int(division_points['right_high_hip'][1] * height))

    left_hip_pixel = (int(division_points['left_hip'][0] * width), int(division_points['left_hip'][1] * height))
    right_hip_pixel = (int(division_points['right_hip'][0] * width), int(division_points['right_hip'][1] * height))

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    edges = cv2.Canny(blurred_img, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def left_boundary(left_pixel):
        for contour in contours:
            for x in range(left_pixel[0], 0, -1):
                if edges[left_pixel[1], x] > 0:
                    left_boundary = (x, left_pixel[1])
                    cv2.circle(img, left_boundary, 5, (0, 255, 0), -1)
                    return left_boundary

    def right_boundary(right_pixel):
        for x in range(right_pixel[0], width):
            if edges[right_pixel[1], x] > 0:
                right_boundary = (x, right_pixel[1])
                cv2.circle(img, right_boundary, 5, (0, 255, 0), -1)
                return right_boundary

    left_shoulder_boundary = left_boundary(left_shoulder_pixel)
    right_shoulder_boundary = right_boundary(right_shoulder_pixel)

    left_waist_boundary = left_boundary(left_waist_pixel)
    right_waist_boundary = right_boundary(right_waist_pixel)

    left_bust_boundary = left_boundary(left_bust_pixel)
    right_bust_boundary = right_boundary(right_bust_pixel)

    left_high_hip_boundary = left_boundary(left_high_hip_pixel)
    right_high_hip_boundary = right_boundary(right_high_hip_pixel)

    left_hip_boundary = left_boundary(left_hip_pixel)
    right_hip_boundary = right_boundary(right_hip_pixel)


    if (left_shoulder_boundary is None or right_shoulder_boundary is None or 
        left_waist_boundary is None or right_waist_boundary is None or
        left_bust_boundary is None or right_bust_boundary is None or
        left_high_hip_boundary is None or right_high_hip_boundary is None or
        left_hip_boundary is None or right_hip_boundary is None):

        raise Exception("The provided photo is not clear enough")
    
    def euclidean_distance(point1, point2):
        return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    shoulder_length = euclidean_distance(left_shoulder_boundary, right_shoulder_boundary)
    bust_length = euclidean_distance(left_bust_boundary, right_bust_boundary)
    waist_length = euclidean_distance(left_waist_boundary, right_waist_boundary)
    high_hip_length = euclidean_distance(left_high_hip_boundary, right_high_hip_boundary)
    hip_length = euclidean_distance(left_hip_boundary, right_hip_boundary)

    def female_body_classifier(bust_length, waist_length, high_hip_length, hip_length):
        percentage_threshold = 0.1
        waist_reduction_threshold = 0.25

        bust_vs_waist = bust_length / waist_length
        hip_vs_bust = hip_length / bust_length
        waist_vs_bust = waist_length / bust_length
        waist_vs_hip = waist_length / hip_length
        shoulder_to_hip_ratio = bust_length / hip_length

        if bust_vs_waist > (1 + percentage_threshold) and bust_length > hip_length and waist_length > hip_length:
            return "Apple Shape"

        elif (waist_vs_bust > (1 - waist_reduction_threshold)) and \
            (bust_length * (1 - percentage_threshold) <= hip_length <= bust_length * (1 + percentage_threshold)):
            return "Rectangle Shape"

        elif (waist_vs_bust < (1 - waist_reduction_threshold)) and \
            (waist_vs_hip < (1 - waist_reduction_threshold)) and \
            (shoulder_to_hip_ratio >= (1 - percentage_threshold) and shoulder_to_hip_ratio <= (1 + percentage_threshold)):
            return "Hourglass Shape"

        elif shoulder_to_hip_ratio > (1 + percentage_threshold):
            return "Inverted Triangle Shape"

        elif hip_vs_bust > (1 + percentage_threshold):
            return "Pear Shape"

        else:
            return "Unclassified Shape"

    def male_body_classifier(shoulder_length, waist_length, hip_length):
        percentage_threshold = 0.2

        shoulder_vs_waist = shoulder_length / waist_length
        shoulder_vs_hip = shoulder_length / hip_length
        waist_vs_hip = waist_length / hip_length

        if (shoulder_length * (1 - percentage_threshold) <= waist_length <= shoulder_length * (1 + percentage_threshold)) and \
            (shoulder_length * (1 - percentage_threshold) <= hip_length <= shoulder_length * (1 + percentage_threshold)):
            return "Rectangle Shape"

        elif shoulder_vs_hip > (1 + percentage_threshold) and waist_length > hip_length:
            return "Inverted Triangle Shape"

        elif shoulder_vs_waist > (1 + percentage_threshold) and waist_vs_hip > (1 - percentage_threshold):
            return "Trapezoid Shape"

        elif waist_length > shoulder_length and waist_length > hip_length:
            return "Oval Shape"

        elif shoulder_vs_hip > (1 + percentage_threshold) and waist_length < shoulder_length and waist_length < hip_length:
            return "Triangle Shape"

        else:
            return "Unclassified Shape"

    if gender == "Men":
        return male_body_classifier(shoulder_length, waist_length, hip_length)
    else:
        return female_body_classifier(bust_length, waist_length, high_hip_length, hip_length)


