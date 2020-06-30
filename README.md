# Mask and SocialDistancing Detection
As the COVID-19 cases are on rise and several preventive steps are being taken up by respective authorities to curb the number of cases.Wearing mask and maintaining social distancing in public places are on top of the pile of prevention measures list.In this prototype I just wanted to demonstrate how can we detect if a person is wearing a mask or not and people are keeping safe distance from others in public places.

Demo: https://youtu.be/_8uE5BBSODU

## step-1
install the required libraries from "requirements.txt".

## step-2
get the dataset from here- https://drive.google.com/drive/folders/10S5jXSG5xwE3VybOKHiZGg0YI6Nt8CKg?usp=sharing

## step-3
create a folder named "images" in the root application directory.Unzip the datasets as "mask" and "nomask" into "images" folder.

## step-4
run "detect_mask_train.py" to train the model for mask detection.

## step-5(Mask Detection)
once training is completed, run "detect_face_mask.py" to test mask detection is working.

## step-6(Social Distancing Detection)
run "social_distance_detector.py" to test the social distance detection.

## step-7(Mask and Social Distance Detection)
run "check_mask_social_distance.py"

See https://www.pyimagesearch.com/2020/06/01/opencv-social-distancing-detector/ to understand how detect distance between objects for social distance detection.
I have used "yolov3-tiny" model for person detection.You can use other "yolov3" models and their configuration files from https://pjreddie.com/darknet/yolo/ .
