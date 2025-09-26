from argparse import ArgumentParser
import numpy as np
import cv2
import mediapipe as mp
import time
import pyvirtualcam

# --- Helper functions for formatted printing ---
def print_header(title):
    """Prints a formatted header to the console."""
    print("\n" + "="*50)
    print(f"|  {title.upper():^44}  |")
    print("="*50)

def print_setting(key, value):
    """Prints a key-value pair with standardized formatting."""
    print(f"  {key:<22}: {value}")

def print_info(message):
    """Prints an informational message."""
    print(f"  > {message}")

def print_success(message):
    """Prints a success message."""
    print(f"  ✓ {message}")

def print_error(message):
    """Prints an error message."""
    print(f"  ✗ {message}")

# --- Argument Parsing ---
parser = ArgumentParser(description="Face-tracking virtual camera using MediaPipe.")
parser.add_argument("-a", "--area", dest="active_area", default=1.2,
                    help="Active area for tracking (e.g., 1.2 for 20%% zoom). Default is 1.2", type=float)
parser.add_argument("-s", "--smooth", dest="average_smooth", default=30, type=int,
                    help="Number of frames to average for smooth tracking. Default is 30.")
parser.add_argument("-ow", "--output_width", dest="preferred_width", default=1280, type=int,
                    help="Preferred output width of the camera. Default value is 1280px.")
parser.add_argument("-oh", "--output_height", dest="preferred_height", default=720, type=int,
                    help="Preferred output height of the camera. Default value is 720px.")
parser.add_argument("-c", "--camera_id", dest="camera_id", default=0, type=int,
                    help="Camera device ID (an integer from 0 to N). Default is 0.")
parser.add_argument("-f", "--fps", dest="camera_fps", default=30, type=int,
                    help="Preferred camera FPS. Default is 30.")
parser.add_argument("-w", "--weights", dest="weight_recent", default=0.3, type=float,
                    help="Weight for recent positions (0.0-1.0). Higher values are more responsive but less smooth.")
parser.add_argument("--no-display", dest="no_display", action="store_true",
                    help="Don't show preview windows (useful for headless operation).")
args = parser.parse_args()


# --- Detect available camera devices ---
print_header("Detecting Camera Devices")
index = 0
available_devices = []
while True:
    cap = cv2.VideoCapture(index)
    if not cap.read()[0]:
        break
    else:
        available_devices.append(index)
    cap.release()
    index += 1

if not available_devices:
    print_error("No camera devices found. Exiting.")
    exit()

print_info(f"Available device IDs: {available_devices}")


# --- Initialize Tracking Variables ---
media_mobile = args.average_smooth
weight_recent = args.weight_recent

mp_face_mesh = mp.solutions.face_mesh

baricentro_x, baricentro_y = [], []
prev_x_face, prev_y_face = None, None
face_detected = False
last_detection_time = 0
prediction_x, prediction_y = 0, 0
velocity_x, velocity_y = 0, 0

# --- Setup Video Capture ---
vc = cv2.VideoCapture(args.camera_id)
if not vc.isOpened():
    raise RuntimeError('Could not open video source')

print_header("Camera Settings")
print_setting("Selected Device ID", args.camera_id)

# Set preferred camera settings
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

print_setting("Applied Resolution", f"{width}x{height}")
print_setting("Applied FPS", int(fps))

# --- Calculate Output & Tracking Settings ---
zoom_scale = args.active_area
width_out = int(width / zoom_scale)
height_out = int(height / zoom_scale)
start_x, start_y = 0, 0
stop_x, stop_y = width_out, height_out

print_header("Tracking & Output Settings")
print_setting("Output Resolution", f"{width_out}x{height_out}")
print_setting("Active Area (Zoom)", f'{1/zoom_scale:.2%}')
print_setting("Smoothing", f"{media_mobile} frames")
print_setting("Recent Pos. Weight", weight_recent)


# --- Helper for Exponential Smoothing ---
def exp_filter(new_val, old_val, alpha):
    if old_val is None:
        return new_val
    return alpha * new_val + (1 - alpha) * old_val


# --- Initialize MediaPipe FaceMesh ---
print_header("MediaPipe FaceMesh Setup")
face_mesh_configs = [
    {'max_num_faces': 1, 'refine_landmarks': True, 'min_detection_confidence': 0.5, 'min_tracking_confidence': 0.5},
    {'max_num_faces': 1, 'refine_landmarks': True, 'min_detection_confidence': 0.7, 'min_tracking_confidence': 0.7},
    {'max_num_faces': 1},
]

face_mesh = None
config_used = None

for i, config in enumerate(face_mesh_configs, 1):
    try:
        print_info(f"Attempting config #{i}: {config}")
        face_mesh = mp_face_mesh.FaceMesh(**config)
        config_used = i
        print_success(f"Configuration #{i} loaded successfully!")
        break
    except Exception as e:
        print_error(f"Configuration #{i} failed: {e}")
        continue

if face_mesh is None:
    print_error("No FaceMesh configuration worked. Try downgrading MediaPipe.")
    print_info("Suggested command: pip install mediapipe==0.9.3.0")
    raise RuntimeError("FaceMesh initialization failed.")


# --- Initialize Virtual Camera ---
print_header("Virtual Camera Setup")
try:
    virtual_cam = pyvirtualcam.Camera(width=width, height=height, fps=int(fps))
    print_success(f"Virtual camera initialized: {width}x{height}@{int(fps)}fps")
    print_setting("Device", virtual_cam.device)
except Exception as e:
    print_error(f"Failed to initialize virtual camera: {e}")
    print_info("Ensure OBS Virtual Camera or similar software is installed and running.")
    raise

print_header("Status")
if not args.no_display:
    print_info('Face tracking active. Press "q" in a preview window to quit.')
else:
    print_info('Face tracking active in headless mode. Press Ctrl+C to quit.')
print("="*50)


# --- Main Loop ---
last_time = time.time()
try:
    while True:
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        ret, image = vc.read()
        if not ret:
            print_error("Failed to grab frame from camera.")
            break

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        face_detected_current_frame = bool(results.multi_face_landmarks)

        if face_detected_current_frame:
            face_detected = True
            last_detection_time = time.time()

            for face_landmarks in results.multi_face_landmarks:
                # Use a central point for more stable tracking
                p1 = face_landmarks.landmark[1] # A point on the nose bridge
                x_face = int(p1.x * width)
                y_face = int(p1.y * height)

                if prev_x_face is not None and dt > 0:
                    velocity_x = (x_face - prev_x_face) / dt
                    velocity_y = (y_face - prev_y_face) / dt
                prev_x_face, prev_y_face = x_face, y_face

                baricentro_x.append(x_face)
                baricentro_y.append(y_face)

            smooth_len = len(baricentro_x)
            if smooth_len > media_mobile:
                baricentro_x.pop(0)
                baricentro_y.pop(0)

                if weight_recent > 0:
                    weights = [(1 + i * weight_recent/media_mobile) for i in range(media_mobile)]
                    total_weight = sum(weights)
                    x_face = int(sum(x * w for x, w in zip(baricentro_x, weights)) / total_weight)
                    y_face = int(sum(y * w for y, w in zip(baricentro_y, weights)) / total_weight)
                else:
                    x_face = int(sum(baricentro_x) / media_mobile)
                    y_face = int(sum(baricentro_y) / media_mobile)
            else:
                x_face = int(sum(baricentro_x) / smooth_len)
                y_face = int(sum(baricentro_y) / smooth_len)

            prediction_x, prediction_y = x_face, y_face

        elif face_detected and time.time() - last_detection_time < 1.0:
            # Predict position for a short time if face is lost
            prediction_x += velocity_x * dt
            prediction_y += velocity_y * dt
            prediction_x = max(0, min(width, prediction_x))
            prediction_y = max(0, min(height, prediction_y))
            x_face, y_face = int(prediction_x), int(prediction_y)

        else:
            if face_detected:
                print_info("Face tracking lost. Centering frame.")
                face_detected = False
            x_face, y_face = width // 2, height // 2

        # Apply exponential smoothing for smoother camera movement
        if not hasattr(args, 'target_x'):
            args.target_x, args.target_y = x_face, y_face

        args.target_x = exp_filter(x_face, args.target_x, 0.1) # Lower alpha = smoother
        args.target_y = exp_filter(y_face, args.target_y, 0.1)

        target_x_int, target_y_int = int(args.target_x), int(args.target_y)

        # Calculate crop window, ensuring it's within bounds
        start_y = max(0, target_y_int - height_out // 2)
        stop_y = min(height, start_y + height_out)
        if stop_y == height: # Adjust start if we hit the bottom
            start_y = height - height_out

        start_x = max(0, target_x_int - width_out // 2)
        stop_x = min(width, start_x + width_out)
        if stop_x == width: # Adjust start if we hit the right edge
            start_x = width - width_out

        # Crop and resize the image
        image_crop = image[start_y:stop_y, start_x:stop_x, :]
        output_frame = cv2.resize(image_crop, (width, height), interpolation=cv2.INTER_CUBIC)

        # Send to virtual camera
        output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        virtual_cam.send(output_frame_rgb)

        if not args.no_display:
            # Draw tracking info on original frame
            color = (0, 255, 0) if face_detected_current_frame else (0, 255, 255)
            cv2.circle(image, (target_x_int, target_y_int), 5, color, -1)
            cv2.rectangle(image, (start_x, start_y), (stop_x, stop_y), color, 2)

            # Display frames
            cv2.imshow('Original with Tracking', image)
            cv2.imshow('Virtual Camera Output', output_frame)

            if cv2.waitKey(1) == ord('q'):
                break

        virtual_cam.sleep_until_next_frame()

except KeyboardInterrupt:
    print("\n" + "="*50)
    print_info("Process interrupted by user (Ctrl+C).")
except Exception as e:
    print_error(f"An unexpected error occurred: {e}")
finally:
    print_header("Shutting Down")
    vc.release()
    print_info("Camera released.")
    if not args.no_display:
        cv2.destroyAllWindows()
    if face_mesh:
        face_mesh.close()
    if 'virtual_cam' in locals() and virtual_cam:
        virtual_cam.close()
        print_info("Virtual camera closed.")
    print_success("Cleanup complete. Exiting.")
