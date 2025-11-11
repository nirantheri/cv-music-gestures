notes

using scikit learn for test train split

Kalman filter for smooth detection
mediapipe
PyTorch or Tensorflow
soundfile, pyfluidsynth, and PyAudio

numpy
opencv

pykalman--> built in kalman filter

unzip the file
both of us will look at the demo.py in hagrid dataset

look at mediapipe and also models and how to create them pytorch

Must use python 3.12 for mediapipe (unsupported later)

eventually create a virtual environment we can both use

practicing pulling images / folders from the dataset using testing.ipynb

setting up a lightweight model?

Model architecture to use
-- lightweight with limited layers
-- taking yolov8
-- R-CNN/Mask R-CNN
https://github.com/Thomas9363/How-to-Train-Custom-Hand-Gestures-Using-Mediapipe/tree/main


all gestures: call, dislike (thumbs down), fist, four (4), grabbing (rawr xd), grip (zero, O), hand_heart (w/ angel wings), hand_heart2 (like the emoji), holy (prayer), like (thumbs up), little_finger (pinkie), middle_finger, mute (shush), no_gesture (hand covering face or holding mug), okay (gotcha), one (erm ackshually), palm (5), peace (2), peace_inverted (genz peace), point, rock (hell yes brother), stop (palm but fingers are together), stop_inverted (back of hand), take_picture (make rectangle w/ both hands), three (3), three_gun (2 finger gun in front of chest), three2 (guy in seminar/alien/asl3), three3 (?), thumb_index (L), thumb_index2 (two Ls), timeout (T), two_up (two fingers together), two_up_inverted, xsign (wakanda 4eva)

list/combine:
one - one + mute
two - peace + two_up
three - three
four - four
five - palm  + stop
zero/reset - fist
synth - three_gun
thumbs up - like
rock - rock

make folder w all the classes/labels we want - copy everything from the folders we want into the ones we do (delete og files we don't want 4 space)

wlocka strat -> make textfile with all the files we dont want and us OS to loop/delete - same thing for moving the files (make)

deleting tests - call is running outside of tmux, rmdir attempt on grabbing