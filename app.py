from argparse import ArgumentParser
import numpy as np
import cv2
import mediapipe as mp
import time
print('------------------------------------')
parser = ArgumentParser()
parser.add_argument("-a", "--area", dest="active_area", default=1.2,
                    help="Active area for tracking. Default is 1.2", type=float)
parser.add_argument("-s", "--smooth", dest="average_smooth", default=30, type=int,
                    help="Smooth tracking taking average position last N points. Default is 30.")
parser.add_argument("-ow", "--output_width", dest="preferred_width", default=1280, type=int,
                    help="Width of the image. Default value is 1280px.")
parser.add_argument("-oh", "--output_height", dest="preferred_height", default=720, type=int,
                    help="Height of the image. Default value is 720px.")
parser.add_argument("-c", "--camera_id", dest="camera_id", default=0, type=int,
                    help="Select camera device ID. An integer from 0 to N. Default is 0.")
parser.add_argument("-f", "--fps", dest="camera_fps", default=30, type=int,
                    help="Select camera device FPS. An integer from 0 to N.")
parser.add_argument("-w", "--weights", dest="weight_recent", default=0.3, type=float,
                    help="Weight for recent positions (0.0-1.0). Higher values make tracking more responsive but less smooth.")
args = parser.parse_args()

# Detect available camera devices
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
weight_recent = args.weight_recent

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For tracking
baricentro_x = []
baricentro_y = []
prev_x_face = None
prev_y_face = None
face_detected = False
last_detection_time = 0
prediction_x = 0
prediction_y = 0
velocity_x = 0
velocity_y = 0

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

print('-- Output Settings (Before Rescale) ----------------------------')
print(f'Height: {height_out}')
print(f'Width: {width_out}')
print(f'Active Area: {round(1/zoom_scale,2)}%')
print(f'Smoothing: {media_mobile} frames with {weight_recent} weight for recent positions')
print()

# Simple exponential filter function
def exp_filter(new_val, old_val, alpha):
    if old_val is None:
        return new_val
    return alpha * new_val + (1 - alpha) * old_val

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    print('Face tracking active. Press "q" to quit.')
    print()

    last_time = time.time()

    while True:
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        ret, image = vc.read()
        if not ret:
            print("Failed to grab frame")
            break

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # calculate height and width.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Track face
        face_detected_current_frame = False

        if results.multi_face_landmarks:
            face_detected_current_frame = True
            face_detected = True
            last_detection_time = time.time()

            for face_landmarks in results.multi_face_landmarks:
                x_face_min = face_landmarks.landmark[234].x
                y_face_min = face_landmarks.landmark[234].y

                x_face_max = face_landmarks.landmark[447].x
                y_face_max = face_landmarks.landmark[447].y

                x_face = int((x_face_max + x_face_min)/2*width)
                y_face = int((y_face_max + y_face_min)/2*height)

                # Update velocities if we have previous positions
                if prev_x_face is not None and dt > 0:
                    velocity_x = (x_face - prev_x_face) / dt
                    velocity_y = (y_face - prev_y_face) / dt

                prev_x_face = x_face
                prev_y_face = y_face

                # Save for weighted moving average
                baricentro_x.append(x_face)
                baricentro_y.append(y_face)

            # Apply weighted smoothing
            smooth = len(baricentro_x)
            if smooth > media_mobile:
                del baricentro_x[0]
                del baricentro_y[0]

                # Enhanced weighted average - more recent points have higher weight
                if weight_recent > 0:
                    weights = [(1 + i * weight_recent/media_mobile) for i in range(media_mobile)]
                    total_weight = sum(weights)
                    x_face = int(sum(x * w for x, w in zip(baricentro_x[-media_mobile:], weights)) / total_weight)
                    y_face = int(sum(y * w for y, w in zip(baricentro_y[-media_mobile:], weights)) / total_weight)
                else:
                    # Standard average
                    x_face = int(sum(baricentro_x[-media_mobile:]) / media_mobile)
                    y_face = int(sum(baricentro_y[-media_mobile:]) / media_mobile)
            else:
                # Simple average for initial frames
                x_face = int(sum(baricentro_x) / smooth)
                y_face = int(sum(baricentro_y) / smooth)

            # Update prediction
            prediction_x = x_face
            prediction_y = y_face
        elif face_detected and time.time() - last_detection_time < 1.0:
            # Face was detected before but not in this frame - use prediction
            prediction_x += velocity_x * dt
            prediction_y += velocity_y * dt

            # Keep predictions within frame boundaries
            prediction_x = max(0, min(width, prediction_x))
            prediction_y = max(0, min(height, prediction_y))

            x_face = int(prediction_x)
            y_face = int(prediction_y)
        else:
            # No face detected for a while, just center the frame
            if face_detected:
                print("Lost face tracking")
                face_detected = False

            x_face = width // 2
            y_face = height // 2

        # Apply exponential smoothing to the target position (smoother transitions)
        if not hasattr(args, 'target_x'):
            args.target_x = x_face
            args.target_y = y_face

        args.target_x = exp_filter(x_face, args.target_x, 0.1)  # Lower alpha = smoother but slower
        args.target_y = exp_filter(y_face, args.target_y, 0.1)

        x_face = int(args.target_x)
        y_face = int(args.target_y)

        # Calculate crop window
        if y_face - height_out/2 < 0:
            start_y = 0
            stop_y = height_out
        elif y_face + height_out/2 > height:
            start_y = height - height_out
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

        # Draw face tracking information
        if face_detected_current_frame:
            cv2.circle(image, (x_face, y_face), 5, (0, 255, 0), -1)
        elif face_detected:
            cv2.circle(image, (x_face, y_face), 5, (0, 255, 255), -1)

        cv2.rectangle(image,
                      (int(start_x), int(start_y)),
                      (int(stop_x), int(stop_y)),
                      (0, 255, 0), 2)

        # Crop the image
        image_crop = image[int(start_y):int(stop_y), int(start_x):int(stop_x), :]

        # Resize for display
        resized = cv2.resize(image_crop, (width, height), interpolation=cv2.INTER_AREA)

        # Display frames (show both original with tracking box and cropped result)
        cv2.imshow('Original with Tracking', image)
        cv2.imshow('Face Tracking Result', resized)

        # Calculate actual FPS
        if hasattr(args, 'frame_count'):
            args.frame_count += 1
        else:
            args.frame_count = 1
            args.fps_time = time.time()

        if args.frame_count % 30 == 0:
            elapsed = time.time() - args.fps_time
            if elapsed > 0:
                current_fps = 30 / elapsed
                cv2.setWindowTitle('Face Tracking Result', f'Face Tracking - {current_fps:.1f} FPS')
                args.fps_time = time.time()

        if cv2.waitKey(1) == ord('q'):
            print("Quit system")
            break

    # Release the capture and close windows
    vc.release()
    cv2.destroyAllWindows()
