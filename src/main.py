import sys
import cv2
import mediapipe as mp
import time
import pdb
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

sys.path.insert(1, '../')
import pykinect_azure as pykinect

if __name__ == "__main__":

	# Initialize the library, if the library is not found, add the library path as argument
	pykinect.initialize_libraries(track_body=True)
	hands = mp_hands.Hands(static_image_mode=True,max_num_hands=2,min_detection_confidence=0.5)

	# Modify camera configuration
	device_config = pykinect.default_configuration
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
	# device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
	# device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
	device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
	#print(device_config)

	# Start device
	device = pykinect.start_device(config=device_config)

	# Start body tracker
	bodyTracker = pykinect.start_body_tracker()
	cap = cv2.VideoCapture(0)
	
	cv2.namedWindow('Color image with skeleton',cv2.WINDOW_NORMAL)

	binary = 1
	while True:
		
		start_ckpt = time.time()
		# Get capture
		capture = device.update()
		dev_update_ckpt = time.time()


		# Get body tracker frame
		body_frame = bodyTracker.update()
		# pdb.set_trace()
		# print(body_frame)
		# quit()
		# if not binary: # is 1
		# 	binary = 0
		# 	continue
		# else:
		# 	binary = 1
		track_update_ckpt = time.time()
		# Get the color image
		ret, image = capture.get_color_image()
		get_color_ckpt = time.time()
		if not ret:
			continue
		



		image.flags.writeable = False
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		prepare_color_ckpt = time.time()
		results = hands.process(image)
		hand_ckpt = time.time()
		image.flags.writeable = True
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

		if results.multi_hand_landmarks:
			for hand_landmarks in results.multi_hand_landmarks:
				mp_drawing.draw_landmarks(image,hand_landmarks,
				mp_hands.HAND_CONNECTIONS,
				mp_drawing_styles.get_default_hand_landmarks_style(),
				mp_drawing_styles.get_default_hand_connections_style())

		color_image = image
 		# Draw the skeletons into the color image
		# color_skeleton = body_frame.draw_bodies(color_image, pykinect.K4A_CALIBRATION_TYPE_COLOR)
		# print(color_image.shape)
		# Overlay body segmentation on depth image
		

		# print(os.path.getsize(file_name)/1024+'KB / '+size+' KB downloaded!', end='\r')


		print(f'Dev: {(dev_update_ckpt-start_ckpt):.2f}s,  Track: {(track_update_ckpt-dev_update_ckpt):.2f}s, Image: { (get_color_ckpt-track_update_ckpt):.2f}s, Prepare: {(prepare_color_ckpt-get_color_ckpt):.2f}s, Hand: {(hand_ckpt- prepare_color_ckpt):.2f}s', end='\r')
		

		cv2.imshow('Color image with skeleton',image)	

		# Press q key to stop
		if cv2.waitKey(1) == ord('q'):  
			break