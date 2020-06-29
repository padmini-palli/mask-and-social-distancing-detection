import cv2
import numpy as np
import os
import imutils
from imutils.video import VideoStream, FileVideoStream, FPS
import time
from operations import predict_image, facedetect
from operations import social_distancing_config as config
from operations.detection import detect_people
from scipy.spatial import distance as dist
from datetime import datetime

#loading face detection model
prototxtPath = "./face_detector/deploy.prototxt"
weightsPath = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

#input video stream
input_video = "./videos/test/test_01.mp4"

#create output dir if not exists for storing recorded videos
output_dir = "./videos/output"
if not os.path.isdir(output_dir):
      os.makedirs(output_dir)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = FileVideoStream(input_video).start() #streaming video from local storage
#vs = VideoStream(src=0).start() ###for webcam or rtsp cam

fps = FPS().start()

writer = None
time.sleep(2.0)

while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	if frame is None:
		break
	frame = imutils.resize(frame, width=1080)

	#detect face and coordinates of face
	(faces, locs) = facedetect.detect_face(frame, faceNet)

	if faces is None:
		pass
	
	#check if face is detected
	elif faces:
		for (face, box) in zip(faces, locs):
			# unpack the bounding box and predictions
			(startX, startY, endX, endY) = box
			#get the prediction of mask
			data = predict_image.main(face)

			text = data[0]["res"]
			percentage = data[1]
			key_list = list(percentage.keys())
			for key, value in percentage.items():
				text = text.title()# Title Case looks Stunning.
				color = (0, 255, 0) if text == "Mask" else (0, 0, 255)
				index = int(key_list.index(key)-1)
				cv2.putText(frame, text, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	#detect people
	results = detect_people(frame)

	if results is None:
		pass

	elif results:

		# initialize the set of indexes that violate the minimum social
		# distance
		violate = set()

		# ensure there are *at least* two people detections (required in
		# order to compute our pairwise distance maps)
		if len(results) >= 2:
			# extract all centroids from the results and compute the
			# Euclidean distances between all pairs of the centroids
			centroids = np.array([r[2] for r in results])
			D = dist.cdist(centroids, centroids, metric="euclidean")

			# loop over the upper triangular of the distance matrix
			for i in range(0, D.shape[0]):
				for j in range(i + 1, D.shape[1]):
					# check to see if the distance between any two
					# centroid pairs is less than the configured number
					# of pixels
					if D[i, j] < config.MIN_DISTANCE:
						# update our violation set with the indexes of
						# the centroid pairs
						violate.add(i)
						violate.add(j)

		# loop over the results
			for (i, (prob, bbox, centroid)) in enumerate(results):
				# extract the bounding box and centroid coordinates, then
				# initialize the color of the annotation
				(startX, startY, endX, endY) = bbox
				(cX, cY) = centroid
				color = (0, 255, 0)

				# if the index pair exists within the violation set, then
				# update the color
				if i in violate:
					color = (0, 0, 255)

				# draw (1) a bounding box around the person and (2) the
				# centroid coordinates of the person,
				cv2.rectangle(frame, (startX-10, startY-25), (endX+10, endY+25), color, 1)
				#cv2.circle(frame, (cX, cY), 5, color, 1)

			# draw the total number of social distancing violations on the
			# output frame
			text_ = "Social Distancing Violations: {}".format(len(violate))
			cv2.putText(frame, text_, (10, frame.shape[0] - 25),
				cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)

	else:
		continue

	# check to see if the output frame should be displayed to our
	# screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	fps.update()

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# if an output video file path has been supplied and the video
	# writer has not been initialized, do so now
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"XVID")
		writer = cv2.VideoWriter(f"{output_dir}/output_{datetime.now().strftime('%Y%m%d%H%M%S')}.avi", fourcc, 25,
			(frame.shape[1], frame.shape[0]), True)

	# if the video writer is not None, write the frame to the output
	# video file
	if writer is not None:
		writer.write(frame)

# do a bit of cleanup
fps.stop()
vs.stop()
cv2.destroyAllWindows()