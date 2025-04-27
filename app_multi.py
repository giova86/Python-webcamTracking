from argparse import ArgumentParser
import numpy as np
import cv2
import mediapipe as mp
import time
import math

parser = ArgumentParser()
parser.add_argument("-a", "--area", dest="active_area", default=1.2,
                    help="Active area for tracking", type=float)
parser.add_argument("-s", "--smooth", dest="average_smooth", default=15, type=int,
                    help="Smooth tracking taking average position last N points. Default is 15.")
parser.add_argument("-ow", "--output_width", dest="preferred_width", default=1280, type=int,
                    help="Width of the image. Default value is 1280px.")
parser.add_argument("-oh", "--output_height", dest="preferred_height", default=720, type=int,
                    help="Height of the image. Default value is 720px.")
parser.add_argument("-c", "--camera_id", dest="camera_id", default=1, type=int,
                    help="Select camera device ID. An integer from 0 to N.")
parser.add_argument("-f", "--fps", dest="camera_fps", default=30, type=int,
                    help="Select camera device FPS. An integer from 0 to N.")
parser.add_argument("-d", "--distance", dest="face_distance_threshold", default=0.3, type=float,
                    help="Distance threshold for separating faces (0.0-1.0). Default is 0.3 (30% of frame width).")
parser.add_argument("-m", "--max_faces", dest="max_faces", default=2, type=int,
                    help="Maximum number of faces to track. Default is 2.")
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
face_distance_threshold = args.face_distance_threshold * args.preferred_width
max_faces = args.max_faces

# Definire uno zoom aggiuntivo per la modalità split
split_zoom_factor = 0.8  # Questo valore determina il livello di zoom aggiuntivo (valori più bassi = più zoom)

mp_face_mesh = mp.solutions.face_mesh

# For tracking multiple faces
class FaceTracker:
    def __init__(self, face_id, buffer_size=15):
        self.face_id = face_id
        self.buffer_size = buffer_size
        self.positions_x = []
        self.positions_y = []
        self.velocity_x = 0
        self.velocity_y = 0
        self.last_seen = time.time()
        self.is_active = True
        self.smoothed_x = None
        self.smoothed_y = None

    def update(self, x, y):
        self.positions_x.append(x)
        self.positions_y.append(y)

        if len(self.positions_x) > self.buffer_size:
            self.positions_x.pop(0)
            self.positions_y.pop(0)

        if len(self.positions_x) >= 2:
            self.velocity_x = self.positions_x[-1] - self.positions_x[-2]
            self.velocity_y = self.positions_y[-1] - self.positions_y[-2]

        self.last_seen = time.time()

        # Update smoothed position
        if len(self.positions_x) > 0:
            self.smoothed_x = sum(self.positions_x) / len(self.positions_x)
            self.smoothed_y = sum(self.positions_y) / len(self.positions_y)
        else:
            self.smoothed_x = x
            self.smoothed_y = y

        return self.smoothed_x, self.smoothed_y

    def predict(self, dt):
        """Predict position based on velocity"""
        if self.smoothed_x is None:
            return None, None

        predicted_x = self.smoothed_x + self.velocity_x
        predicted_y = self.smoothed_y + self.velocity_y

        return predicted_x, predicted_y

    def get_position(self):
        if self.smoothed_x is None:
            return None, None
        return self.smoothed_x, self.smoothed_y

# Initialize face trackers
face_trackers = {}

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

print('-- Output Settings ----------------------------')
print(f'Height: {height_out}')
print(f'Width: {width_out}')
print(f'Active Area: {round(1/zoom_scale,2)}%')
print(f'Max Faces: {max_faces}')
print(f'Face Distance Threshold: {face_distance_threshold}px')
print(f'Split View Zoom Factor: {split_zoom_factor}')
print()

# Simple exponential filter function
def exp_filter(new_val, old_val, alpha):
    if old_val is None:
        return new_val
    return alpha * new_val + (1 - alpha) * old_val

# Calculate distance between two points
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Function to find and assign face IDs
def find_closest_face(face_positions, trackers):
    assigned = set()
    new_trackers = {}
    current_time = time.time()

    # First, try to match faces with existing trackers
    for pos_idx, (x, y) in enumerate(face_positions):
        closest_id = None
        min_distance = float('inf')

        for tracker_id, tracker in trackers.items():
            if tracker_id in assigned or (current_time - tracker.last_seen) > 1.0:
                continue

            tracker_x, tracker_y = tracker.get_position()
            if tracker_x is None:
                continue

            distance = calculate_distance(x, y, tracker_x, tracker_y)
            if distance < min_distance:
                min_distance = distance
                closest_id = tracker_id

        # If we found a match, update the tracker
        if closest_id is not None and min_distance < width * 0.25:  # Match if within 25% of frame width
            trackers[closest_id].update(x, y)
            assigned.add(closest_id)
            new_trackers[closest_id] = trackers[closest_id]
        else:
            # Create a new tracker for this face
            new_id = len(trackers) + 1
            while new_id in trackers:
                new_id += 1
            new_tracker = FaceTracker(new_id, buffer_size=media_mobile)
            new_tracker.update(x, y)
            new_trackers[new_id] = new_tracker
            assigned.add(new_id)

    # Keep recently active trackers that weren't updated
    for tracker_id, tracker in trackers.items():
        if tracker_id not in assigned and (current_time - tracker.last_seen) < 1.0:
            new_trackers[tracker_id] = tracker

    # Limit to max_faces trackers, keeping most recently seen
    if len(new_trackers) > max_faces:
        sorted_trackers = sorted(new_trackers.items(),
                                key=lambda x: x[1].last_seen,
                                reverse=True)
        new_trackers = {k: v for k, v in sorted_trackers[:max_faces]}

    return new_trackers

def crop_around_point(image, center_x, center_y, crop_width, crop_height, frame_width, frame_height):
    """Crop image around a center point with proper boundary handling"""
    if center_y - crop_height/2 < 0:
        start_y = 0
        stop_y = crop_height
    elif center_y + crop_height/2 > frame_height:
        start_y = frame_height - crop_height
        stop_y = frame_height
    else:
        start_y = center_y - crop_height/2
        stop_y = center_y + crop_height/2

    if center_x - crop_width/2 < 0:
        start_x = 0
        stop_x = crop_width
    elif center_x + crop_width/2 > frame_width:
        start_x = frame_width - crop_width
        stop_x = frame_width
    else:
        start_x = center_x - crop_width/2
        stop_x = center_x + crop_width/2

    return image[int(start_y):int(stop_y), int(start_x):int(stop_x)], (int(start_x), int(start_y), int(stop_x), int(stop_y))

with mp_face_mesh.FaceMesh(
    max_num_faces=max_faces,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    print(f'Multi-face tracking active. Tracking up to {max_faces} faces. Press "q" to quit.')
    print()

    last_time = time.time()
    last_mode_switch = time.time()
    current_mode = "unified"  # "unified" or "split"

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

        # Calculate height and width
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Original image for drawing
        original_image = image.copy()

        # Track faces
        face_positions = []

        if results.multi_face_landmarks:
            for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
                # Get face center using nose landmark
                nose_tip = face_landmarks.landmark[1]
                x_face = int(nose_tip.x * width)
                y_face = int(nose_tip.y * height)
                face_positions.append((x_face, y_face))

                # Draw face landmark
                cv2.circle(original_image, (x_face, y_face), 5, (0, 255, 0), -1)

        # Update face trackers
        face_trackers = find_closest_face(face_positions, face_trackers)

        # Check distances between faces to determine display mode
        should_split = False
        tracked_faces = []

        for tracker_id, tracker in face_trackers.items():
            x, y = tracker.get_position()
            if x is not None:
                tracked_faces.append((tracker_id, x, y))

        # Sort by x-position for consistent left/right assignment
        tracked_faces.sort(key=lambda f: f[1])

        # Determine if we should split the view
        if len(tracked_faces) >= 2:
            # Calculate average distance between adjacent faces
            total_distance = 0
            face_count = len(tracked_faces)

            for i in range(face_count - 1):
                _, x1, y1 = tracked_faces[i]
                _, x2, y2 = tracked_faces[i+1]
                total_distance += calculate_distance(x1, y1, x2, y2)

            avg_distance = total_distance / (face_count - 1)

            # If average distance exceeds threshold, use split view
            if avg_distance > face_distance_threshold:
                should_split = True

        # Prevent frequent mode switching by using hysteresis
        if should_split != (current_mode == "split"):
            if current_time - last_mode_switch > 1.0:  # Require 1 second stability before switching
                current_mode = "split" if should_split else "unified"
                last_mode_switch = current_time
                print(f"Switching to {current_mode} view")

        # Create output image(s)
        if current_mode == "unified" or len(tracked_faces) < 2:
            # Unified view - calculate bounding box that includes all faces
            if tracked_faces:
                # Get centroid of all faces
                centroid_x = sum(x for _, x, _ in tracked_faces) / len(tracked_faces)
                centroid_y = sum(y for _, _, y in tracked_faces) / len(tracked_faces)

                # Crop image around centroid
                image_crop, (start_x, start_y, stop_x, stop_y) = crop_around_point(
                    image, centroid_x, centroid_y, width_out, height_out, width, height)

                # Draw bounding box on original image
                cv2.rectangle(original_image,
                            (start_x, start_y),
                            (stop_x, stop_y),
                            (0, 255, 0), 2)

                # Resize for display
                resized = cv2.resize(image_crop, (width, height), interpolation=cv2.INTER_AREA)
                # cv2.putText(resized, "Unified View", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                #             1, (0, 255, 0), 2, cv2.LINE_AA)

                # Display both images
                cv2.imshow('Multi-Face Tracking - Original', original_image)
                cv2.imshow('Multi-Face Tracking - Result', resized)
            else:
                # No faces detected, show the full frame
                cv2.putText(original_image, "No faces detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Multi-Face Tracking - Original', original_image)
                cv2.imshow('Multi-Face Tracking - Result', image)
        else:
            # Split view - create separate frames for each face with enhanced zoom
            split_image = np.zeros((height, width, 3), dtype=np.uint8)

            # Get the left and right faces (we're limiting to 2 faces for split view)
            if len(tracked_faces) > 2:
                # If more than 2 faces, take the leftmost and rightmost
                left_face = tracked_faces[0]
                right_face = tracked_faces[-1]
            else:
                left_face = tracked_faces[0]
                right_face = tracked_faces[1]

            # Get positions
            left_id, left_x, left_y = left_face
            right_id, right_x, right_y = right_face

            # Calculate appropriate crop sizes for split view with additional zoom
            # We divide by 2 for split view and then apply additional zoom factor
            split_width_out = int(width_out // 2 * split_zoom_factor)
            split_height_out = int(height_out * split_zoom_factor)

            # Individually crop around each face center with enhanced zoom
            left_crop, (left_start_x, left_start_y, left_stop_x, left_stop_y) = crop_around_point(
                image, left_x, left_y, split_width_out, split_height_out, width, height)

            right_crop, (right_start_x, right_start_y, right_stop_x, right_stop_y) = crop_around_point(
                image, right_x, right_y, split_width_out, split_height_out, width, height)

            # Draw bounding boxes on original image
            cv2.rectangle(original_image,
                        (left_start_x, left_start_y),
                        (left_stop_x, left_stop_y),
                        (0, 255, 0), 2)
            cv2.rectangle(original_image,
                        (right_start_x, right_start_y),
                        (right_stop_x, right_stop_y),
                        (0, 0, 255), 2)

            # Resize for display - maintain aspect ratio
            half_width = width // 2
            # Calculate height to maintain aspect ratio
            display_height = int((split_height_out / split_width_out) * half_width)

            # If display_height is larger than frame height, adjust both dimensions
            if display_height > height:
                scale = height / display_height
                display_height = height
                display_width = int(half_width * scale)
                # Center the crops horizontally
                left_offset = (half_width - display_width) // 2
                right_offset = (half_width - display_width) // 2
            else:
                display_width = half_width
                left_offset = 0
                right_offset = 0

            # Resize crops to maintain aspect ratio
            left_resized = cv2.resize(left_crop, (display_width, display_height), interpolation=cv2.INTER_AREA)
            right_resized = cv2.resize(right_crop, (display_width, display_height), interpolation=cv2.INTER_AREA)

            # Create background for split view
            split_image = np.zeros((height, width, 3), dtype=np.uint8)

            # Calculate vertical offset to center the crops vertically
            vert_offset = (height - display_height) // 2

            # Place resized crops in the split image
            if vert_offset >= 0:
                split_image[vert_offset:vert_offset+display_height, left_offset:left_offset+display_width] = left_resized
                split_image[vert_offset:vert_offset+display_height, half_width+right_offset:half_width+right_offset+display_width] = right_resized
            else:
                # If the image is too tall, crop it to fit
                crop_start = -vert_offset
                crop_height = display_height - 2*crop_start
                split_image[:, left_offset:left_offset+display_width] = left_resized[crop_start:crop_start+crop_height, :]
                split_image[:, half_width+right_offset:half_width+right_offset+display_width] = right_resized[crop_start:crop_start+crop_height, :]

            # Add dividing line
            cv2.line(split_image, (half_width, 0), (half_width, height), (255, 255, 255), 2)

            # Add labels
            # cv2.putText(split_image, f"Face {left_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            #             1, (0, 255, 0), 2, cv2.LINE_AA)
            # cv2.putText(split_image, f"Face {right_id}", (half_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            #             1, (0, 0, 255), 2, cv2.LINE_AA)

            # Display both images
            cv2.imshow('Multi-Face Tracking - Original', original_image)
            cv2.imshow('Multi-Face Tracking - Result', split_image)

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
                cv2.setWindowTitle('Multi-Face Tracking - Result', f'Face Tracking - {current_fps:.1f} FPS')
                args.fps_time = time.time()

        if cv2.waitKey(1) == ord('q'):
            print("Quit system")
            break

    # Release the capture and close windows
    vc.release()
    cv2.destroyAllWindows()
