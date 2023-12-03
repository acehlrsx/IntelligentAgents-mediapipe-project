import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Video capture
cap = cv2.VideoCapture(0)

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Function to calculate the Euclidean distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        # Take only the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # Get the coordinates of the thumb tip (Landmark 4) and index finger tip (Landmark 8)
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # Calculate the distance between thumb tip and index finger tip
        distance = calculate_distance(thumb_tip, index_finger_tip)

        # Move the mouse cursor (inverted coordinates)
        inverted_x = int((1 - index_finger_tip.x) * screen_width)
        inverted_y = int(index_finger_tip.y * screen_height)
        pyautogui.moveTo(inverted_x, inverted_y, duration=0.1)

        # Check if the hand is closed (distance below a threshold)
        if distance < 0.05:  # Adjust the threshold as needed
            # Simulate a mouse click
            pyautogui.click()

        # Display the distance on the frame
        cv2.putText(frame, f'Distance: {distance:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Flip the frame horizontally for the mirror effect
    frame = cv2.flip(frame, 1)

    # Display the frame
    cv2.imshow('Hand Tracking', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and destroy OpenCV windows
cap.release()
cv2.destroyAllWindows()
