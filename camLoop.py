import cv2
import mediapipe as mp
import numpy as np
import time
import camDetectionFunctions
import nithFunctions
import udpFunctions

# Initialize the face mesh model.
mp_face_mesh = mp.solutions.face_mesh

# Create a face mesh object with minimum detection confidence and minimum tracking confidence thresholds.
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.1,  # Minimum confidence value for the face detection model
    min_tracking_confidence=0.1,  # Minimum confidence value for the landmark tracking model)
)
# Import the drawing utilities module from Mediapipe.
mp_drawing = mp.solutions.drawing_utils

# Define the specifications for drawing the face mesh annotations.
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


def get_available_cameras():
    """Detect available cameras and return a list of working camera indexes with faster timeout."""
    available_cameras = []

    # Try potential camera indexes (typically 0-9 covers most setups)
    for i in range(10):
        # Set a short timeout for non-existent cameras
        cap = cv2.VideoCapture(i)

        # Set a small timeout (in milliseconds)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 300)

        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"Camera {i} is available")
        cap.release()

    return available_cameras


def select_camera():
    """Let the user select a camera from the available ones."""
    available_cameras = get_available_cameras()

    if not available_cameras:
        print("No cameras detected!")
        return 0

    if len(available_cameras) == 1:
        print(
            f"Only one camera detected (index {available_cameras[0]}). Using this camera."
        )
        return available_cameras[0]

    print("\nAvailable cameras:")
    for i, cam_index in enumerate(available_cameras):
        print(f"{i+1}. Camera {cam_index}")

    selection = 0
    while True:
        try:
            selection = int(input(f"Select a camera (1-{len(available_cameras)}): ")) - 1
            if 0 <= selection < len(available_cameras):
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a number.")

    return available_cameras[selection]


# Select camera and create a video capture object
selected_camera = select_camera()
print(f"Using camera {selected_camera}")
cap = cv2.VideoCapture(selected_camera)


def main_loop():
    while cap.isOpened():
        start = time.time()
        success, image = cap.read()

        # Flip image horizontally
        # Convert space from BGR to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance
        image.flags.writeable = False

        # Get result
        results = face_mesh.process(image)

        # Improve performance again
        image.flags.writeable = True

        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape

        # If there are results
        if results.multi_face_landmarks:
            # Get first face
            face_landmarks = results.multi_face_landmarks[0]

            # Get orientation
            x, y, z = camDetectionFunctions.get_face_orientation(
                face_landmarks, img_w, img_h
            )

            # Get eyes aperture ratio
            # using eye aperture ratio
            # (
            #     left_eye_aperture_ratio,
            #     right_eye_aperture_ratio,
            # ) = camDetectionFunctions.get_eye_aperture_ratio_SEGMENTSMETHOD(
            #     face_landmarks
            # )

            # using areas
            # (
            #     left_eye_aperture_ratio,
            #     right_eye_aperture_ratio,
            # ) = camDetectionFunctions.get_eye_aperture_ratio_SEGMENTSMETHOD(
            #     face_landmarks
            # )

            # using twosegmentsmethod
            (
                left_eye_aperture_ratio,
                right_eye_aperture_ratio,
            ) = camDetectionFunctions.get_eye_aperture_ratio_TWOSEGMENTSMETHOD(
                face_landmarks
            )

            # Get mouth aperture ratio
            mouth_aperture_ratio = camDetectionFunctions.get_mouth_aperture_ratio(
                face_landmarks
            )

            # Get head roll
            head_roll = camDetectionFunctions.get_head_roll(face_landmarks)

            # Get eyebrows
            eyebrow_left = camDetectionFunctions.get_left_eyebrow_aperture_ratio(face_landmarks)
            eyebrow_right = camDetectionFunctions.get_right_eyebrow_aperture_ratio(face_landmarks)
            eyebrow_weirdaverage = camDetectionFunctions.get_eyebrows_ratio_2(face_landmarks)

            # region Face Detection Metrics (Top Left)
            # Eyes section
            eyes_text = f"Eyes - L: {np.round(left_eye_aperture_ratio, 2):.2f} | R: {np.round(right_eye_aperture_ratio, 2):.2f}"
            (text_width, text_height), _ = cv2.getTextSize(eyes_text, cv2.FONT_HERSHEY_PLAIN, 0.8, 1)
            cv2.rectangle(image, (5, 5), (15 + text_width, 25 + text_height), (64, 64, 64), -1)
            cv2.putText(
                image,
                eyes_text,
                (10, 20),
                cv2.FONT_HERSHEY_PLAIN,
                0.8,
                (255, 255, 255),
                1,
            )
            
            # Mouth section
            mouth_text = f"Mouth: {np.round(mouth_aperture_ratio, 2):.2f}"
            (text_width, text_height), _ = cv2.getTextSize(mouth_text, cv2.FONT_HERSHEY_PLAIN, 0.8, 1)
            cv2.rectangle(image, (5, 25), (15 + text_width, 45 + text_height), (64, 64, 64), -1)
            cv2.putText(
                image,
                mouth_text,
                (10, 40),
                cv2.FONT_HERSHEY_PLAIN,
                0.8,
                (255, 255, 255),
                1,
            )
            
            # Eyebrows section
            eyebrows_text = f"Eyebrows - L: {np.round(eyebrow_left, 2):.2f} | R: {np.round(eyebrow_right, 2):.2f} | Avg: {np.round(eyebrow_weirdaverage, 2):.2f}"
            (text_width, text_height), _ = cv2.getTextSize(eyebrows_text, cv2.FONT_HERSHEY_PLAIN, 0.8, 1)
            cv2.rectangle(image, (5, 45), (15 + text_width, 65 + text_height), (64, 64, 64), -1)
            cv2.putText(
                image,
                eyebrows_text,
                (10, 60),
                cv2.FONT_HERSHEY_PLAIN,
                0.8,
                (255, 255, 255),
                1,
            )
            # endregion

            # region Head Orientation (Top Right)
            orientation_title = f"Head Orientation"
            (text_width, text_height), _ = cv2.getTextSize(orientation_title, cv2.FONT_HERSHEY_PLAIN, 0.8, 1)
            cv2.rectangle(image, (img_w - 205, 5), (img_w - 5, 25 + text_height), (64, 64, 64), -1)
            cv2.putText(
                image,
                orientation_title,
                (img_w - 200, 20),
                cv2.FONT_HERSHEY_PLAIN,
                0.8,
                (0, 255, 255),
                1,
            )
            
            yaw_text = f"Yaw: {np.round(x, 2):.2f}"
            (text_width, text_height), _ = cv2.getTextSize(yaw_text, cv2.FONT_HERSHEY_PLAIN, 0.8, 1)
            cv2.rectangle(image, (img_w - 205, 25), (img_w - 5, 45 + text_height), (64, 64, 64), -1)
            cv2.putText(
                image,
                yaw_text,
                (img_w - 200, 40),
                cv2.FONT_HERSHEY_PLAIN,
                0.8,
                (0, 255, 255),
                1,
            )
            
            pitch_text = f"Pitch: {np.round(y, 2):.2f}"
            (text_width, text_height), _ = cv2.getTextSize(pitch_text, cv2.FONT_HERSHEY_PLAIN, 0.8, 1)
            cv2.rectangle(image, (img_w - 205, 45), (img_w - 5, 65 + text_height), (64, 64, 64), -1)
            cv2.putText(
                image,
                pitch_text,
                (img_w - 200, 60),
                cv2.FONT_HERSHEY_PLAIN,
                0.8,
                (0, 255, 255),
                1,
            )
            
            roll_text = f"Roll: {np.round(head_roll, 2):.2f}"
            (text_width, text_height), _ = cv2.getTextSize(roll_text, cv2.FONT_HERSHEY_PLAIN, 0.8, 1)
            cv2.rectangle(image, (img_w - 205, 65), (img_w - 5, 85 + text_height), (64, 64, 64), -1)
            cv2.putText(
                image,
                roll_text,
                (img_w - 200, 80),
                cv2.FONT_HERSHEY_PLAIN,
                0.8,
                (0, 255, 255),
                1,
            )
            # endregion

            # region Performance (Bottom Left)
            end = time.time()
            totalTime = end - start
            fps = 1 / totalTime

            performance_text = f"Performance: {int(fps)} FPS"
            (text_width, text_height), _ = cv2.getTextSize(performance_text, cv2.FONT_HERSHEY_PLAIN, 0.8, 1)
            cv2.rectangle(image, (5, img_h - 55), (15 + text_width, img_h - 25), (64, 64, 64), -1)
            cv2.putText(
                image,
                performance_text,
                (10, img_h - 40),
                cv2.FONT_HERSHEY_PLAIN,
                0.8,
                (0, 255, 0),
                1,
            )
            # endregion

            # region Connection (Bottom Right)
            connection_text = f"UDP Port: {udpFunctions.PORT}"
            (text_width, text_height), _ = cv2.getTextSize(connection_text, cv2.FONT_HERSHEY_PLAIN, 0.8, 1)
            cv2.rectangle(image, (img_w - 125, img_h - 55), (img_w - 5, img_h - 25), (64, 64, 64), -1)
            cv2.putText(
                image,
                connection_text,
                (img_w - 120, img_h - 40),
                cv2.FONT_HERSHEY_PLAIN,
                0.8,
                (0, 255, 0),
                1,
            )
            # endregion

            # region Controls (Bottom Center)
            controls_text = "Press ESC to exit"
            (text_width, text_height), _ = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_PLAIN, 0.7, 1)
            cv2.rectangle(image, (img_w // 2 - 65, img_h - 35), (img_w // 2 + 65, img_h - 5), (64, 64, 64), -1)
            cv2.putText(
                image,
                controls_text,
                (img_w // 2 - 60, img_h - 20),
                cv2.FONT_HERSHEY_PLAIN,
                0.7,
                (255, 255, 255),
                1,
            )
            # endregion

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                landmark_drawing_spec=drawing_spec,
            )

            nithFunctions.send_data(
                x * 3,
                y * 3,
                head_roll,
                mouth_aperture_ratio,
                left_eye_aperture_ratio,
                right_eye_aperture_ratio,
            )

        cv2.imshow("NITHwebcamWrapper", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
