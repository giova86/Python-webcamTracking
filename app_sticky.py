import cv2
import pyvirtualcam
from pyvirtualcam import PixelFormat
import mediapipe as mp
import numpy as np
from utils import mediapipe_detection, draw_landmarks, draw_landmarks_custom, draw_limit_rh, draw_limit_lh, check_detection

index = 0
arr = []
while True:
    cap = cv2.VideoCapture(index)
    if not cap.read()[0]:
        break
    else:
        arr.append(index)
    cap.release()
    index += 1
print(arr)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
#mp_drawing_styles = mp.solutions.drawing_styles
#mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

logo = cv2.imread('logo.png', cv2.IMREAD_GRAYSCALE)
logo = cv2.flip(logo, 1)
scale_percent = 50 # percent of original size
width = int(logo.shape[1] * scale_percent / 100)
height = int(logo.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
logo = cv2.resize(logo, dim, interpolation = cv2.INTER_AREA)


vc = cv2.VideoCapture(1)

if not vc.isOpened():
    raise RuntimeError('Could not open video source')

pref_width = 1280
pref_height = 720
pref_fps = 30
vc.set(cv2.CAP_PROP_FRAME_WIDTH, pref_width)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, pref_height)
vc.set(cv2.CAP_PROP_FPS, pref_fps)

# Query final capture device values
# (may be different from preferred settings)
width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = vc.get(cv2.CAP_PROP_FPS)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    with pyvirtualcam.Camera(width, height, fps, fmt=PixelFormat.BGR) as cam:
        print('Virtual camera device: ' + cam.device)
        while True:
            ret, image = vc.read()

            image, results2 = mediapipe_detection(image, holistic)

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #results = face_mesh.process(image)
            image_shape = image.shape

            img = np.zeros((image_shape[0], image_shape[1], 3), np.uint8)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            draw_landmarks_custom(img, results2)

            img[(image_shape[0]-dim[1]-42):(image_shape[0]-42), (image_shape[1]-dim[0]-42):(image_shape[1]-42),:]=logo.reshape(dim[1],dim[0],1)

            # cv2.imshow('check video', img)
            cam.send(img)
            cam.sleep_until_next_frame()
