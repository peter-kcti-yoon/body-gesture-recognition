import cv2
import mediapipe as mp
import time
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


# For webcam input:
cap = cv2.VideoCapture()
cap.open(0)

assert cap.isOpened(), 'Camera is not accessable.'
hands = mp_hands.Hands(
    model_complexity=0,
    max_num_hands=4,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
poses = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5)

while cap.isOpened():
  success, image = cap.read()
  if not success:
    print("Ignoring empty camera frame.")
    # If loading a video, use 'break' instead of 'continue'.
    continue

  # To improve performance, optionally mark the image as not writeable to
  # pass by reference.
  image.flags.writeable = False
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  start_ckpt = time.time()
  hadns_res = hands.process(image)
  pose_res = poses.process(image)
  print('took ', time.time()-start_ckpt, end='\r')
  # Draw the hand annotations on the image.
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  # if hadns_res.multi_hand_landmarks:
  #   for hand_landmarks in hadns_res.multi_hand_landmarks:
  #     mp_drawing.draw_landmarks(
  #         image,
  #         hand_landmarks,
  #         mp_hands.HAND_CONNECTIONS,
  #         mp_drawing_styles.get_default_hand_landmarks_style(),
  #         mp_drawing_styles.get_default_hand_connections_style())

  # if pose_res.pose_landmarks:
  #     mp_drawing.draw_landmarks(
  #     image,
  #     pose_res.pose_landmarks,
  #     mp_pose.POSE_CONNECTIONS,
  #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
  # Flip the image horizontally for a selfie-view display.
  cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
  if cv2.waitKey(1) & 0xFF == 27:
    break
cap.release()