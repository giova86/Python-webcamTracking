from argparse import ArgumentParser
import numpy as np
import cv2
import mediapipe as mp
import pyvirtualcam
from pyvirtualcam import PixelFormat

parser = ArgumentParser()
parser.add_argument("-a", "--area", dest="active_area", default=1.2,
                    help="Active area for tracking", type=float)
parser.add_argument("-s", "--smooth", dest="average_smooth", default=60, type=int,
                    help="Smooth tracking taking average position last N points. Default is 60.")
parser.add_argument("-ow", "--output_width", dest="preferred_width", default=1280, type=int,
                    help="Width of the image. Default value is 1280px.")
parser.add_argument("-oh", "--output_height", dest="preferred_height", default=720, type=int,
                    help="Height of the image. Default value is 720px.")
parser.add_argument("-c", "--camera_id", dest="camera_id", default=1, type=int,
                    help="Select camera device ID. An integer from 0 to N.")
parser.add_argument("-f", "--fps", dest="camera_fps", default=10, type=int,
                    help="Select camera device FPS. An integer from 0 to N.")
args = parser.parse_args()

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

media_mobile = args.average_smooth

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

baricentro_x = []
baricentro_y = []

vc = cv2.VideoCapture(args.camera_id)

if not vc.isOpened():
    raise RuntimeError('Could not open video source')

print('-- Camera Settings -----------------------------------')
print(f'Device ID: {args.camera_id}')
print()


print('-- Camera Settings -----------------------------------')
print(f'Height: {int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))}')
print(f'Width: {int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))}')
print(f'FPS: {int(vc.get(cv2.CAP_PROP_FPS))}')
print()

pref_width = args.preferred_width
pref_height = args.preferred_height
pref_fps = args.camera_fps
vc.set(cv2.CAP_PROP_FRAME_WIDTH, pref_width)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, pref_height)
vc.set(cv2.CAP_PROP_FPS, pref_fps)

# Query final capture device values
# (may be different from preferred settings)
width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = vc.get(cv2.CAP_PROP_FPS)

print('-- Original Output Settings -----------------------------------')
print(f'Height: {height}')
print(f'Width: {width}')
print(f'FPS: {int(fps)}')
print()

zoom_scale = args.active_area
width_out = int(width/zoom_scale)
height_out = int(height/zoom_scale)

start_x = 0
start_y = 0
stop_x = width_out
stop_y = height_out

print('-- Output Settings (Before Rescale ----------------------------')
print(f'Height: {height_out}')
print(f'Width: {width_out}')
print(f'Active Area: {round(1/zoom_scale,2)}%')
print()

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    # with pyvirtualcam.Camera(width, height, fps, fmt=PixelFormat.BGR) as cam:
    # with pyvirtualcam.Camera(width, height, fps, fmt=PixelFormat.BGR) as cam:
    #     print()
    #     print('Virtual camera device: ' + cam.device)
    #     print()

    while True:
        ret, image = vc.read()

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # calculate height and width.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                x_face_min = face_landmarks.landmark[234].x
                y_face_min = face_landmarks.landmark[234].y

                x_face_max = face_landmarks.landmark[447].x
                y_face_max = face_landmarks.landmark[447].y

                x_face = int((x_face_max + x_face_min)/2*width)
                y_face = int((y_face_max + y_face_min)/2*height)

            baricentro_x.append(x_face)
            baricentro_y.append(y_face)

            smooth = len(baricentro_x)

            if smooth > media_mobile:
                del baricentro_x[0]
                del baricentro_y[0]
                x_face = int(sum(baricentro_x)/media_mobile)
                y_face = int(sum(baricentro_y)/media_mobile)
            else:
                x_face = int(sum(baricentro_x)/smooth)
                y_face = int(sum(baricentro_y)/smooth)

            if y_face - height_out/2 < 0:
                start_y = 0
                stop_y = height_out
            elif y_face + height_out/2 > height:
                start_y =  height - height_out
                stop_y = height
            else:
                start_y = y_face-height_out/2
                stop_y = y_face+height_out/2

            if x_face - width_out/2 < 0:
                start_x = 0
                stop_x = width_out
            elif x_face + width_out/2 > width:
                start_x = width - width_out
                stop_x = width
            else:
                start_x = x_face - width_out/2
                stop_x = x_face + width_out/2

            image_crop = image[int(start_y):int(stop_y), int(start_x):int(stop_x), :]

        else:
            image_crop = image[int(start_y):int(stop_y), int(start_x):int(stop_x), :]

        cv2.imshow('check video2', image_crop)          # Scommenta per testare
        # resized = cv2.resize(image_crop, (width, height), interpolation = cv2.INTER_AREA)
        # cam.send(resized)
        # cam.sleep_until_next_frame()
        if cv2.waitKey(1) == ord('q'):
            print("Quit system")
            break
        # if cv2.waitKey(1) == ord('z'):
        #     tracking = False
        #     print("Tracking OFF")                      # To be done
        # if cv2.waitKey(1) == ord('x'):
        #     tracking = True
        #     print("Tracking ON")                       # To be done
