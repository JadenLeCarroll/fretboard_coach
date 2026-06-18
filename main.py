import cv2
import mediapipe as mp
import numpy as np
import math

print("Starting Fretboard Coach: Stabilized Engine...")

# Initialize MediaPipe's drawing and hand tracking modules
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Grab the default webcam feed
cap = cv2.VideoCapture(0)

# Dictionary defining the target grid matrix for chords
# Only track the physical fingers pressing down, not open strings
CHORD_DICTIONARIES = {
    "A Minor": [
        {"string": 3, "fret": 2, "label": "M"}, # D string, 2nd fret
        {"string": 2, "fret": 2, "label": "R"}, # G string, 2nd fret
        {"string": 1, "fret": 1, "label": "I"}  # B string, 1st fret
    ],
    "C Major": [
        {"string": 4, "fret": 3, "label": "R"}, # A string, 3rd fret  
        {"string": 3, "fret": 2, "label": "M"}, # D string, 2nd fret
        {"string": 1, "fret": 1, "label": "I"}  # B string, 1st fret
    ]
}

ACTIVE_CHORD = "A Minor"
target_notes = CHORD_DICTIONARIES[ACTIVE_CHORD]

# Set up mathematical 2D matrix
calibration_points = []
flat_w, flat_h = 500, 200
fret_width = flat_w // 5
string_height = flat_h // 5

# Smoothing engine preventing the tracking dots from jittering by blending new and old frames
previous_positions = {"I": None, "M": None, "R": None}
smoothing = 0.6  # 0.0 is zero smoothing, 0.9 is extreme lag

def calculate_joint_angle(a, b, c):
    """
    Calculates the 3D interior angle between three joints.
    Used to figure out if a finger is hovering flat or arched to press a string.
    """
    # Create vectors between the joints
    ba = [a.x - b.x, a.y - b.y, a.z - b.z]
    bc = [c.x - b.x, c.y - b.y, c.z - b.z]
    
    # Calculate the dot product and magnitudes
    dot_product = sum(i*j for i, j in zip(ba, bc))
    mag_ba = math.sqrt(sum(i**2 for i in ba))
    mag_bc = math.sqrt(sum(i**2 for i in bc))
    
    # Prevent division by zero if vectors are exactly on top of each other
    if mag_ba * mag_bc == 0: 
        return 0
        
    return math.degrees(math.acos(dot_product / (mag_ba * mag_bc)))

def select_points(event, x, y, flags, param):
    """Mouse callback to grab the 4 corners of the guitar neck for calibration."""
    global calibration_points
    if event == cv2.EVENT_LBUTTONDOWN:
        # Reset the array if the user starts a new 4-point calibration cycle
        if len(calibration_points) == 4:
            calibration_points = []
        calibration_points.append((x, y))

# Set up the UI window and attach our mouse listener
cv2.namedWindow('Fretboard Coach')
cv2.setMouseCallback('Fretboard Coach', select_points)

# Boot up the hand tracking model with strict confidence thresholds
with mp_hands.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.75) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success: 
            continue

        # Flip the image like a mirror for a better user experience
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        h, w, c = image.shape
        
        # Pass the frame to the AI
        results = hands.process(image)
        
        # Convert back to OpenCV's color format for rendering UI
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw the user's calibration clicks
        for pt in calibration_points:
            cv2.circle(image, pt, 5, (0, 165, 255), -1)
        
        # If we have 4 points, build the Homography transform matrix
        homography_matrix = None
        if len(calibration_points) == 4:
            cv2.polylines(image, [np.array(calibration_points)], True, (0, 165, 255), 2)
            pts_src = np.float32(calibration_points)
            pts_dst = np.float32([[0, 0], [flat_w, 0], [flat_w, flat_h], [0, flat_h]])
            # This matrix un-warps the camera perspective back to our flat 2D grid
            homography_matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

        satisfied_targets = set()

        # If the AI sees a hand, process the kinematics
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                # Bundle the specific joints we care about for each finger
                fingers_to_track = [
                    (mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.INDEX_FINGER_TIP, "I"),
                    (mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, "M"),
                    (mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_TIP, "R")
                ]
                
                for mcp_idx, pip_idx, tip_idx, label in fingers_to_track:
                    mcp = hand_landmarks.landmark[mcp_idx]
                    pip = hand_landmarks.landmark[pip_idx]
                    tip = hand_landmarks.landmark[tip_idx]
                    
                    # Convert AI normalized coordinates (0.0 - 1.0) to actual screen pixels
                    raw_cx, raw_cy = int(tip.x * w), int(tip.y * h)
                    
                    # Apply our exponential moving average (EMA) to smooth out the tracking
                    if previous_positions[label] is None:
                        cx, cy = raw_cx, raw_cy
                    else:
                        px, py = previous_positions[label]
                        cx = int((raw_cx * (1 - smoothing)) + (px * smoothing))
                        cy = int((raw_cy * (1 - smoothing)) + (py * smoothing))
                    
                    previous_positions[label] = (cx, cy)
                    
                    # Kinematic Gate: Ensure the finger is arched (< 150 degrees)
                    bend_angle = calculate_joint_angle(mcp, pip, tip)
                    is_pressing = bend_angle < 150.0 
                    
                    # Turn the tracking dot green if they are pressing properly
                    finger_color = (0, 255, 0) if is_pressing else (0, 0, 255)
                    cv2.circle(image, (cx, cy), 10, finger_color, cv2.FILLED)
                    
                    # Map the physical coordinate to our matrix using the homography engine
                    if is_pressing and homography_matrix is not None:
                        pt_3d = np.float32([[[cx, cy]]])
                        pt_flat = cv2.perspectiveTransform(pt_3d, homography_matrix)
                        flat_x, flat_y = pt_flat[0][0]
                        
                        # Check for collision inside our string/fret boundaries
                        if 0 <= flat_x <= flat_w and 0 <= flat_y <= flat_h:
                            fret_idx = int(flat_x // fret_width)
                            string_idx = 5 - int(flat_y // string_height) 
                            
                            # Validate against the active chord requirements
                            for i, target in enumerate(target_notes):
                                if target["string"] == string_idx and target["fret"] == fret_idx and target["label"] == label:
                                    satisfied_targets.add(i)

        # Render the master status UI box if the system is calibrated
        if homography_matrix is not None:
            all_pressed = len(satisfied_targets) == len(target_notes)
            status = f"{ACTIVE_CHORD}: VALIDATED" if all_pressed else f"{ACTIVE_CHORD}: INCOMPLETE ({len(satisfied_targets)}/{len(target_notes)})"
            color = (0, 255, 0) if all_pressed else (0, 0, 255)
            
            cv2.rectangle(image, (20, 20), (450, 70), (0, 0, 0), cv2.FILLED)
            cv2.putText(image, status, (35, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Fretboard Coach', image)
        
        # Listen for keyboard interrupts
        key = cv2.waitKey(5) & 0xFF
        if key == 27: # Esc key closes the app
            break
        elif key == ord('c'): # 'c' key clears the homography calibration
            calibration_points = []

# Clean up system resources
cap.release()
cv2.destroyAllWindows()