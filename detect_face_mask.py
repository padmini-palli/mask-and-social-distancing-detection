import cv2
import numpy as np
import os
import imutils
from imutils.video import VideoStream, FPS
import time
from operations import predict_image, facedetect

print("[INFO] loading face detector model...")
prototxtPath = "./face_detector/deploy.prototxt"
weightsPath = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

fps = FPS().start()

time.sleep(2.0)

while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame=cv2.flip(frame,1,0)
	frame = imutils.resize(frame, width=700)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(faces, locs) = facedetect.detect_face(frame, faceNet)
	for (face, box) in zip(faces, locs):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		data = predict_image.main(face)

		text = data[0]["res"]
		percentage = data[1]
		key_list = list(percentage.keys())
		for key, value in percentage.items():
			text = text.title()# Title Case looks Stunning.
			color = (0, 255, 0) if text == "Mask" else (0, 0, 255)
			index = int(key_list.index(key)-1)
			cv2.putText(frame, text, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			cv2.rectangle(frame, (100, index * 20 + 40), (100 +int(float(value)), (index + 1) * 20 + 4),
			                        (255, 0, 0), -1)
			cv2.putText(frame, key, (10, index * 20 + 40),
			                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (7, 109, 16), 2)
			cv2.putText(frame, value, (105 + int(float(value)), index * 20 + 40),
			                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	fps.update()

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
fps.stop()
vs.stop()
cv2.destroyAllWindows()