import cv2
import numpy as np

def detect_laser_beam(video_path):
    # Open video capture
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the range of the laser color in HSV
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])

        # Create a binary mask for the laser color
        mask = cv2.inRange(hsv_frame, lower_red, upper_red)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original frame
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

        # Display the result
        cv2.imshow('Laser Detection', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'path_to_your_video.mp4'  # Replace with the path to your video file
    detect_laser_beam(video_path)

