from math import dist

import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates

# cam = cv2.VideoCapture(0)
# face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
# screen_w, screen_h = pyautogui.size()
# while True:
#     _, frame = cam.read()
#     frame = cv2.flip(frame, 1)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     output = face_mesh.process(rgb_frame)
#     landmark_points = output.multi_face_landmarks
#     frame_h, frame_w, _ = frame.shape
#     if landmark_points:
#         landmarks = landmark_points[0].landmark
#         # for id, landmark in enumerate(landmarks[474:478]):
#         for id, landmark in enumerate(landmarks):
#             x = int(landmark.x * frame_w)
#             y = int(landmark.y * frame_h)
#             cv2.circle(frame, (x, y), 3, (0, 255, 0))
#             if id == 1:
#                 screen_x = screen_w / frame_w * x
#                 screen_y = screen_h / frame_h * y
#                 pyautogui.moveTo(screen_x, screen_y)
#             left = [landmarks[145], landmarks[159]]
#             for landmark in left:
#                 x = int(landmark.x * frame_w)
#                 y = int(landmark.y * frame_h)
#                 cv2.circle(frame, (x, y), 3, (0, 255, 0))
#             if (left[0].y - left[1].y) < 0.004:
#                 print('click')
#                 pyautogui.sleep(0)
#     cv2.imshow('Eye Controlled Mouse', frame)
#     cv2.waitKey(1)


# //////////////////////////////////////////////////////////

# def get_mediapipe_app(
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5,
# ):
#     """Initialize and return Mediapipe FaceMesh Solution Graph object"""
#     face_mesh = mp.solutions.face_mesh.FaceMesh(
#         max_num_faces=max_num_faces,
#         refine_landmarks=refine_landmarks,
#         min_detection_confidence=min_detection_confidence,
#         min_tracking_confidence=min_tracking_confidence,
#     )
#
#     return face_mesh
#
#
# def distance(point_1, point_2):
#     """Calculate l2-norm between two points"""
#     dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
#     return dist
#
#
# def get_ear(landmarks, refer_idxs, frame_width, frame_height):
#     """
#     Calculate Eye Aspect Ratio for one eye.
#     Args:
#         landmarks: (list) Detected landmarks list
#         refer_idxs: (list) Index positions of the chosen landmarks
#                             in order P1, P2, P3, P4, P5, P6
#         frame_width: (int) Width of captured frame
#         frame_height: (int) Height of captured frame
#     Returns:
#         ear: (float) Eye aspect ratio
#     """
#     try:
#         # Compute the euclidean distance between the horizontal
#         coords_points = []
#         for i in refer_idxs:
#             lm = landmarks[i]
#             coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
#             coords_points.append(coord)
#
#         # Eye landmark (x, y)-coordinates
#         P2_P6 = distance(coords_points[1], coords_points[5])
#         P3_P5 = distance(coords_points[2], coords_points[4])
#         P1_P4 = distance(coords_points[0], coords_points[3])
#
#         # Compute the eye aspect ratio
#         ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)
#
#     except:
#         ear = 0.0
#         coords_points = None
#
#     return ear, coords_points
#
#
# def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
#     # Calculate Eye aspect ratio
#
#     left_ear, left_lm_coordinates = get_ear(landmarks, left_eye_idxs, image_w, image_h)
#     right_ear, right_lm_coordinates = get_ear(landmarks, right_eye_idxs, image_w, image_h)
#     Avg_EAR = (left_ear + right_ear) / 2.0
#
#     return Avg_EAR, (left_lm_coordinates, right_lm_coordinates)
#
#
# def plot_eye_landmarks(frame, left_lm_coordinates, right_lm_coordinates, color):
#     for lm_coordinates in [left_lm_coordinates, right_lm_coordinates]:
#         if lm_coordinates:
#             for coord in lm_coordinates:
#                 cv2.circle(frame, coord, 2, color, -1)
#
#     frame = cv2.flip(frame, 1)
#     return frame
#
#
# def plot_text(image, text, origin, color, font=cv2.FONT_HERSHEY_SIMPLEX, fntScale=0.8, thickness=2):
#     image = cv2.putText(image, text, origin, font, fntScale, color, thickness)
#     return image
#
#
# class VideoFrameHandler:
#     def __init__(self):
#         """
#         Initialize the necessary constants, mediapipe app
#         and tracker variables
#         """
#         # Left and right eye chosen landmarks.
#         self.eye_idxs = {
#             "left": [362, 385, 387, 263, 373, 380],
#             "right": [33, 160, 158, 133, 153, 144],
#         }
#
#         # Used for coloring landmark points.
#         # Its value depends on the current EAR value.
#         self.RED = (0, 0, 255)  # BGR
#         self.GREEN = (0, 255, 0)  # BGR
#
#         # Initializing Mediapipe FaceMesh solution pipeline
#         self.facemesh_model = get_mediapipe_app()
#
#         # For tracking counters and sharing states in and out of callbacks.
#         self.state_tracker = {
#             "start_time": time.perf_counter(),
#             "DROWSY_TIME": 0.0,  # Holds the amount of time passed with EAR < EAR_THRESH
#             "COLOR": self.GREEN,
#             "play_alarm": False,
#         }
#
#         self.EAR_txt_pos = (10, 30)
#
#     def process(self, frame: np.array, thresholds: dict):
#         """
#         This function is used to implement our Drowsy detection algorithm
#         Args:
#             frame: (np.array) Input frame matrix.
#             thresholds: (dict) Contains the two threshold values
#                                WAIT_TIME and EAR_THRESH.
#         Returns:
#             The processed frame and a boolean flag to
#             indicate if the alarm should be played or not.
#         """
#
#         # To improve performance,
#         # mark the frame as not writeable to pass by reference.
#         frame.flags.writeable = False
#         frame_h, frame_w, _ = frame.shape
#
#         DROWSY_TIME_txt_pos = (10, int(frame_h // 2 * 1.7))
#         ALM_txt_pos = (10, int(frame_h // 2 * 1.85))
#
#         results = self.facemesh_model.process(frame)
#
#         if results.multi_face_landmarks:
#             landmarks = results.multi_face_landmarks[0].landmark
#             EAR, coordinates = calculate_avg_ear(landmarks, self.eye_idxs["left"], self.eye_idxs["right"], frame_w, frame_h)
#             frame = plot_eye_landmarks(frame, coordinates[0], coordinates[1], self.state_tracker["COLOR"])
#
#             if EAR < thresholds["EAR_THRESH"]:
#
#                 # Increase DROWSY_TIME to track the time period with EAR less than the threshold
#                 # and reset the start_time for the next iteration.
#                 end_time = time.perf_counter()
#
#                 self.state_tracker["DROWSY_TIME"] += end_time - self.state_tracker["start_time"]
#                 self.state_tracker["start_time"] = end_time
#                 self.state_tracker["COLOR"] = self.RED
#
#                 if self.state_tracker["DROWSY_TIME"] >= thresholds["WAIT_TIME"]:
#                     self.state_tracker["play_alarm"] = True
#                     plot_text(frame, "WAKE UP! WAKE UP", ALM_txt_pos, self.state_tracker["COLOR"])
#
#             else:
#                 self.state_tracker["start_time"] = time.perf_counter()
#                 self.state_tracker["DROWSY_TIME"] = 0.0
#                 self.state_tracker["COLOR"] = self.GREEN
#                 self.state_tracker["play_alarm"] = False
#
#             EAR_txt = f"EAR: {round(EAR, 2)}"
#             DROWSY_TIME_txt = f"DROWSY: {round(self.state_tracker['DROWSY_TIME'], 3)} Secs"
#             plot_text(frame, EAR_txt, self.EAR_txt_pos, self.state_tracker["COLOR"])
#             plot_text(frame, DROWSY_TIME_txt, DROWSY_TIME_txt_pos, self.state_tracker["COLOR"])
#
#         else:
#             self.state_tracker["start_time"] = time.perf_counter()
#             self.state_tracker["DROWSY_TIME"] = 0.0
#             self.state_tracker["COLOR"] = self.GREEN
#             self.state_tracker["play_alarm"] = False
#
#             # Flip the frame horizontally for a selfie-view display.
#             frame = cv2.flip(frame, 1)
#
#         return frame, self.state_tracker["play_alarm"]

# //////////////////////////////////////////////////////////////////////////////////////////////////

# import cv2
# import mediapipe as mp
#
# mp_drawing = mp.solutions.drawing_utils
# mp_face_detection = mp.solutions.face_detection
# mp_face_mesh = mp.solutions.face_mesh
#
# # Initialize face detection and face mesh models
# face_detection = mp_face_detection.FaceDetection()
# face_mesh = mp_face_mesh.FaceMesh()
#
# cap = cv2.VideoCapture(0)
#
#
# def classify_eye(left_eye_roi):
#     pass
#
#
# def extract_eye_roi(roi, left_eye_landmarks):
#     pass
#
#
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     # Run face detection
#     results = face_detection.process(frame)
#
#     # Extract face region
#     if results.detections:
#         for detection in results.detections:
#             bbox = detection.location_data.relative_bounding_box
#             x, y, w, h = int(bbox.xmin * frame.shape[1]), int(bbox.ymin * frame.shape[0]), \
#                          int(bbox.width * frame.shape[1]), int(bbox.height * frame.shape[0])
#             roi = frame[y:y+h, x:x+w]
#
#             # Run face mesh on ROI to get eye landmarks
#             results = face_mesh.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
#             if results.multi_face_landmarks:
#                 for face_landmarks in results.multi_face_landmarks:
#                     # Extract eye landmarks
#                     # left_eye_landmarks = face_landmarks.landmark[mp_face_mesh.FACE_LANDMARKS_LEFT_EYE]
#                     left_eye_landmarks = face_landmarks.landmark[456]
#                     # right_eye_landmarks = face_landmarks.landmark[mp_face_mesh.FACE_LANDMARKS_RIGHT_EYE]
#                     right_eye_landmarks = face_landmarks.landmark[234]
#
#                     # Apply eye classification model to left and right eyes
#                     left_eye_roi = extract_eye_roi(roi, left_eye_landmarks)
#                     left_eye_status = classify_eye(left_eye_roi)
#
#                     right_eye_roi = extract_eye_roi(roi, right_eye_landmarks)
#                     right_eye_status = classify_eye(right_eye_roi)
#
#                     # Draw bounding boxes around eyes
#                     # mp_drawing.draw_landmarks(roi, left_eye_landmarks, mp_face_mesh.FACE_CONNECTIONS)
#                     mp_drawing.draw_landmarks(481, 480, 456)
#                     # mp_drawing.draw_landmarks(roi, right_eye_landmarks, mp_face_mesh.FACE_CONNECTIONS)
#                     mp_drawing.draw_landmarks(266, 220, 265)
#
#                     # Display eye status
#                     cv2.putText(roi, f"Left eye: {left_eye_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                     cv2.putText(roi, f"Right eye: {right_eye_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#             # Display ROI
#             cv2.imshow('ROI', roi)

######################################################################################################

import cv2 as cv2
import numpy as np
import mediapipe as mp
from pygame import mixer

mixer.init()
mixer.music.load("music.wav")




def open_len(arr):
    y_arr = []

    for _,y in arr:
        y_arr.append(y)

    min_y = min(y_arr)
    max_y = max(y_arr)

    return max_y - min_y




mp_face_mesh = mp.solutions.face_mesh

# A: location of the eye-landmarks in the face-mesh collection
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
MOUTH = [61, 146, 91, 181, 84, 17, 314, 321, 375, 291, 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 191]
# MOUTH = [321, 375, 291, 61, 185, 40, 39, 37, 267, 14, 317, 402, 318, 324, 78, 191,80, 81]

# handle of the webcam
cap = cv2.VideoCapture(0)

# Mediapipe parameters
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)as face_mesh:

    # B: count how many frames the user seems to be going to nap (half closed eyes)
    drowsy_frames = 0

    # C: max height of each eye
    max_left = 0
    max_right = 0

    while True:

        # get every frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Get the current frame and collect the image information
        framme= cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]

        # D: collect the mediapipe results
        results = face_mesh.process(rgb_frame)

        # E: if mediapipe was able to find any landmarks in the frame...
        if results.multi_face_landmarks:

            # F: collect all [x,y] pairs of all facial landmarks
            all_landmarks = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])


            # G: right and left eye landmarks
            right_eye = all_landmarks[RIGHT_EYE]
            left_eye = all_landmarks[LEFT_EYE]
            mouth = all_landmarks[MOUTH]

            # H: draw only landmarks of the eyes over the image
            cv2.polylines(frame, [left_eye], True, (0,255,0), 1, cv2.LINE_AA)
            cv2.polylines(frame, [right_eye], True, (0,255,0), 1, cv2.LINE_AA)
            cv2.polylines(frame, [mouth], True, (0,255,0), 1, cv2.LINE_AA)

            # I: estimate eye-height for each eye
            len_left = open_len(right_eye)
            len_right = open_len(left_eye)
            len_mouth = open_len(mouth)

            len_eye = len_right + len_left

            # J: keep the highest distance of eye-height for each eye
            if len_left > max_left:
                # max_left = len_left
                max_left = 10
            #
            if len_right > max_right:
                # max_right = len_right
                max_right = 10
            #
            # if len_mouth < len_eye:
            #     len_eye = len_mouth

            # print on screen the eye-height for each eye
            cv2.putText(img=frame, text='Max: ' + str(max_left) + ' Left Eye: ' + str(len_left), fontFace=0, org=(10, 30), fontScale=0.5, color=(0, 255, 0))
            cv2.putText(img=frame, text='Max: ' + str(max_right) + ' Right Eye: ' + str(len_right), fontFace=0, org=(10, 50), fontScale=0.5, color=(0, 255, 0))
            # cv2.putText(img=frame, text='Max: ' + str(len_eye) + ' Mouth: ' + str(len_mouth), fontFace=0, org=(10, 70), fontScale=0.5, color=(0, 255, 0))

            # K: condition: if eyes are half-open the count.
            if (len_left <= int(max_left / 2) + 1 and len_right <= int(max_right / 2) + 1):
                drowsy_frames += 1
            else:
                drowsy_frames = 0

            # L: if count is above k, that means the person has drowsy eyes for more than k frames.
            if (drowsy_frames > 20):
                cv2.putText(img=frame, text='ALERT' , fontFace=0, org=(200, 300), fontScale=3, color=(0, 255, 0), thickness =3)
                mixer.music.play()


        cv2.imshow('img', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()