import cv2
import mediapipe as mp

print("Starting Fretboard Coach... Loading modules.")
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

print("Connecting to webcam...")
# Change (0) to (1) if it grabs an iPhone Continuity Camera instead of the built-in one
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Could not open camera. Try changing VideoCapture(0) to (1).")
    exit()

print("✅ Camera active! Looking for hands... (Press 'Esc' in the video window to quit)")

with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Flip horizontally for a mirror effect and convert color space for MediaPipe
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        
        # Run the hand tracking model
        results = hands.process(image)

        # Convert back to BGR for OpenCV rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw the skeletal wireframe if hands are found
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the video window
        cv2.imshow('Fretboard Coach - Tracker', image)
        
        # Listen for the 'Esc' key to close cleanly
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Clean up processes
cap.release()
cv2.destroyAllWindows()
print("Process terminated cleanly.")