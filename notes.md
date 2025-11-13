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

Use scipy train_test_split instead of manually splitting?
pip install split-folders will allow us to create folders for train/test/val split instead

creating our own model vs using a pretrained one
we might try to create our own model first and then compare it to a pretrained model

we need to create our own virtual environment in the cs153 folder not in our own folders
mediapipe

use prerecorded videos for heavier models on the server since we can't access video

video capture could be done with mac library called "metal" which is a shared mem between cpu/gpu