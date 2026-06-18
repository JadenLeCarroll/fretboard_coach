import cv2
import mediapipe as mp

print("Starting Fretboard Matrix Engine...")
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

# Define target chords using an array tracking: (string_index, fret_index)
# Strings are 0 to 5 (Low E to High e). Frets are 0 to 4 (Open space to 4th fret).
CHORD_DICTIONARIES = {
    "A Minor": [
        {"string": 1, "fret": 0, "label": "A-Open"}, # A string open
        {"string": 3, "fret": 2, "label": "Middle"}, # D string, 2nd fret
        {"string": 2, "fret": 2, "label": "Ring"},   # G string, 2nd fret
        {"string": 1, "fret": 1, "label": "Index"}   # B string, 1st fret
    ],
    "C Major": [
        {"string": 4, "fret": 3, "label": "Ring"},   # A string, 3rd fret
        {"string": 3, "fret": 2, "label": "Middle"}, # D string, 2nd fret
        {"string": 1, "fret": 1, "label": "Index"}   # B string, 1st fret
    ]
}

# Select target chord to track
ACTIVE_CHORD_NAME = "A Minor"
target_notes = CHORD_DICTIONARIES[ACTIVE_CHORD_NAME]

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        h, w, c = image.shape
        
        # 1. Define the Master Bounding Box for the Guitar Neck
        # Position this box so your physical guitar neck aligns inside it
        neck_x, neck_y = 100, 150
        neck_w, neck_h = 450, 200
        
        # Calculate individual spacing for 5 frets and 6 strings
        fret_width = neck_w // 5
        string_height = neck_h // 5
        
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Keep track of which targets are currently being pressed
        satisfied_targets = set()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 2. Extract Key Fingertips
                fingertips = {
                    "Index": hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                    "Middle": hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                    "Ring": hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                }
                
                # 3. Process Each Finger Coordinate
                for name, landmark in list(fingertips.items()):
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    
                    # Highlight fingertip
                    cv2.circle(image, (cx, cy), 10, (255, 255, 0), cv2.FILLED)
                    
                    # Check if finger is within the overall neck boundary
                    if neck_x < cx < neck_x + neck_w and neck_y < cy < neck_y + neck_h:
                        # Convert pixel delta to matrix index
                        fret_idx = (cx - neck_x) // fret_width
                        string_idx = 5 - ((cy - neck_y) // string_height) # Invert so Low E is bottom
                        
                        # Verify against target notes for active chord
                        for i, target in enumerate(target_notes):
                            if target["string"] == string_idx and target["fret"] == fret_idx:
                                satisfied_targets.add(i)

        # 4. Draw the Dynamic UI Matrix Grid
        # Draw Fret Lines (Vertical)
        for f in range(6):
            fx = neck_x + (f * fret_width)
            cv2.line(image, (fx, neck_y), (fx, neck_y + neck_h), (200, 200, 200), 2)
            if f > 0:
                cv2.putText(image, f"Fret {f}", (fx - fret_width + 10, neck_y + neck_h + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Draw String Lines (Horizontal)
        string_labels = ["e", "B", "G", "D", "A", "E"]
        for s in range(6):
            sy = neck_y + (s * string_height)
            cv2.line(image, (neck_x, sy), (neck_x + neck_w, sy), (150, 150, 150), 1 + (5 - s)//2)
            cv2.putText(image, string_labels[s], (neck_x - 20, sy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

        # 5. Overlay Target Indicators and Evaluation Status
        for i, target in enumerate(target_notes):
            # Calculate center coordinate of target string/fret cell
            tx = neck_x + (target["fret"] * fret_width) + (fret_width // 2)
            ty = neck_y + ((5 - target["string"]) * string_height) + (string_height // 2)
            
            is_pressed = i in satisfied_targets
            color = (0, 255, 0) if is_pressed else (0, 0, 255)
            
            cv2.circle(image, (tx, ty), 12, color, cv2.FILLED)
            cv2.putText(image, target["label"], (tx - 15, ty - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw Master Status Box
        all_pressed = len(satisfied_targets) == len(target_notes)
        status_text = f"{ACTIVE_CHORD_NAME}: VALIDATED" if all_pressed else f"{ACTIVE_CHORD_NAME}: INCOMPLETE"
        status_color = (0, 255, 0) if all_pressed else (0, 0, 255)
        
        cv2.rectangle(image, (30, 30), (350, 80), (0, 0, 0), cv2.FILLED)
        cv2.putText(image, status_text, (45, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        cv2.imshow('Fretboard Coach - Matrix Engine', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()