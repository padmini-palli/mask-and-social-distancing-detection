from operations import social_distancing_config as config
from operations.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
from imutils.video import VideoStream, FileVideoStream, FPS
import imutils
import cv2

input_video = "./videos/test/pedestrians.mp4"

# initialize the video stream
print("[INFO] accessing video stream...")
vs = FileVideoStream(input_video).start() #streaming video from local storage
#vs = VideoStream(src=0).start() ###for webcam or rtsp cam

fps = FPS().start()

# loop over the frames from the video stream
while True:
	# read the next frame from the file
	frame = vs.read()

	# if the frame is none, then we have reached the end
	# of the stream
	if frame is None:
		break

	# resize the frame and then detect people (and only people) in it
	frame = imutils.resize(frame, width=700)

	results = detect_people(frame)

	# initialize the set of indexes that violate the minimum social
	# distance
	if results:
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
			# (cX, cY) = centroid
			color = (0, 255, 0)

			# if the index pair exists within the violation set, then
			# update the color
			if i in violate:
				color = (0, 0, 255)

			# draw (1) a bounding box around the person
			cv2.rectangle(frame, (startX-10, startY-25), (endX+10, endY+25), color, 1)

		# draw the total number of social distancing violations on the
		# output frame
		text = "Social Distancing Violations: {}".format(len(violate))
		cv2.putText(frame, text, (10, frame.shape[0] - 25),
			cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	fps.update()

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
#release vs and destroy all opened windows
fps.stop()
vs.stop()
cv2.destroyAllWindows()