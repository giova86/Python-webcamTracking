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

print('------------------------------------')
print('Devices')
print('------------------------------------')
print(arr)
print()

# width_cropped = width_out
# height_cropped = height_out
media_mobile = 100

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

baricentro_x = []
baricentro_y = []

vc = cv2.VideoCapture(1)

if not vc.isOpened():
    raise RuntimeError('Could not open video source')

print('-- Camera Settings -----------------------------------')
print(f'Height: {int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))}')
print(f'Width: {int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))}')
print(f'FPS: {int(vc.get(cv2.CAP_PROP_FPS))}')
print()

# pref_width = 1280
# pref_height = 720
# pref_fps = 30

# vc.set(cv2.CAP_PROP_FRAME_WIDTH, pref_width)
# vc.set(cv2.CAP_PROP_FRAME_HEIGHT, pref_height)
# vc.set(cv2.CAP_PROP_FPS, pref_fps)

# Query final capture device values
# (may be different from preferred settings)
width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = vc.get(cv2.CAP_PROP_FPS)

# with mp_holistic.Holistic(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as holistic:

zoom_scale = 1.5
width_out = int(width/zoom_scale)
height_out = int(height/zoom_scale)

start_x = 0
start_y = 0
stop_x = width_out
stop_y = height_out

print('-- Output Settings -----------------------------------')
print(f'Height: {height_out}')
print(f'Width: {width_out}')
print(f'Scale Factor: {zoom_scale}')
print()

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    # with pyvirtualcam.Camera(width, height, fps, fmt=PixelFormat.BGR) as cam:
    with pyvirtualcam.Camera(width_out, height_out, fps, fmt=PixelFormat.BGR) as cam:
        print()
        print('Virtual camera device: ' + cam.device)
        print()

        while True:
            ret, image = vc.read()

            # image, results2 = mediapipe_detection(image, holistic)

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
            image_shape = image.shape

            #img = np.zeros((image_shape[0], image_shape[1], 3), np.uint8)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    x_face_min = face_landmarks.landmark[234].x
                    y_face_min = face_landmarks.landmark[234].y

                    x_face_max = face_landmarks.landmark[447].x
                    y_face_max = face_landmarks.landmark[447].y

                    x_face = (x_face_max + x_face_min)/2
                    y_face = (y_face_max + y_face_min)/2

                    x_face=int(x_face*image.shape[1])
                    y_face=int(y_face*image.shape[0])

                    baricentro_x.append(x_face)
                    baricentro_y.append(y_face)

                    if len(baricentro_x) > media_mobile:
                        del baricentro_x[0]
                        del baricentro_y[0]
                        x_face = int(sum(baricentro_x)/media_mobile)
                        y_face = int(sum(baricentro_y)/media_mobile)
                    else:
                        x_face = int(sum(baricentro_x)/len(baricentro_x))
                        y_face = int(sum(baricentro_y)/len(baricentro_y))

                    if y_face - height_out/2 < 0:
                        start_y = 0
                        stop_y = height_out
                    elif y_face + height_out/2 > image.shape[0]:
                        start_y =  image.shape[0] - height_out
                        stop_y = image.shape[0]
                    else:
                        start_y = y_face-height_out/2
                        stop_y = y_face+height_out/2

                    if x_face - width_out/2 < 0:
                        start_x = 0
                        stop_x = width_out
                    elif x_face + width_out/2 > image.shape[1]:
                        start_x = image.shape[1] - width_out
                        stop_x = image.shape[1]
                    else:
                        start_x = x_face - width_out/2
                        stop_x = x_face + width_out/2

                    image_crop = image[int(start_y):int(stop_y), int(start_x):int(stop_x), :]

            else:
                image_crop = image[int(start_y):int(stop_y), int(start_x):int(stop_x), :]

            cv2.imshow('check video2', image_crop)          # Scommenta per testare
            cam.send(image_crop)
            cam.sleep_until_next_frame()
            if cv2.waitKey(33) == ord('q'):
                print("Quit system")
                break
