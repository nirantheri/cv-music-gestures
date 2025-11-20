import mediapipe as mp
import cv2
import os
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype")

# Requires lower python version (3.10)
outermost_folder = r"/mnt/cs/cs153/datasets/music_prod_gestures"

gesture_folders = os.listdir(outermost_folder)

dest_folder = r"/mnt/cs/cs153/datasets/cropped_gestures"

model_path = r'/mnt/home/nvithiananthan/courses/cs153/cv-music-gestures/hand_landmarker.task'

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options,
                                    num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)
exceptions = []

for folder in tqdm(gesture_folders):
    source_path = os.path.join(outermost_folder, folder)

    dest_path = os.path.join(dest_folder, folder)
    os.makedirs(dest_path, exist_ok = True)

    imgs = os.listdir(source_path)

    for img_path in tqdm(imgs, leave=False):
        full_path = os.path.join(source_path, img_path)

        try:
            output_path = os.path.join(dest_path, img_path)


            # Load the input image.
            image = mp.Image.create_from_file(full_path)
            output_ndarray = image.numpy_view()
            output_img = np.copy(output_ndarray)

            height = image.height
            width = image.width

            # Detect hand landmarks from the input image.
            detection_result = detector.detect(image)

            if len(detection_result.hand_landmarks) != 1:
                continue
            else:
                # go to the landmarks in the image and find min/max x and min/max y
                hand_landmarks = detection_result.hand_landmarks[0]
                xs = [lm.x * width  for lm in hand_landmarks]
                ys = [lm.y * height for lm in hand_landmarks]

                left_x, right_x = min(xs), max(xs)
                top_y, bottom_y = min(ys), max(ys)

                # Grow bounding box by 20%
                w = right_x - left_x
                h = bottom_y - top_y

                pad_w = 0.3 * w
                pad_h = 0.3 * h

                left_x -= pad_w
                right_x += pad_w
                top_y -= pad_h
                bottom_y += pad_h

                left_x = max(0, int(left_x))
                right_x = min(width, int(right_x))
                top_y = max(0, int(top_y))
                bottom_y = min(height, int(bottom_y))

                cropped = output_img[top_y:bottom_y, left_x:right_x]

                cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, cropped_bgr)

        except Exception as e:
            exceptions.append(full_path)
            continue

file_name = "list_errors.txt"

with open(file_name, "w") as file:
    for item in exceptions:
        file.write(item + "\n") # Writes each item on a new line
