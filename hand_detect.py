import cv2
import torch
from torchvision import transforms

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# Load the pre-trained model
import os
import json
import sys
# sys.path.append(os.path.abspath(r'C:\Users\niran\VScode\cs153\final_project\cv-music-gestures\hagrid_models'))  # Ensure the path is absolute and correct
# from hagrid_models.models.resnet import ResNet

# with open("classes.json", "r") as f:
#     idx_to_class = json.load(f)

# num_classes = 34
# model = ResNet(num_classes=num_classes)
# model.load_state_dict(torch.load("resnet18.pth", map_location="cpu"))
# model.eval()

# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),  
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

model_path = r'hand_landmarker.task'

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options,
                                    num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)
exceptions = []

def main():
    cap = cv2.VideoCapture(0)

    print("Press 'q' to exit the video stream.")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # # Preprocess the frame for the model
        # input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # input_tensor = transform(input_frame).unsqueeze(0)

        # # Perform inference
        # with torch.no_grad():
        #     logits = model(input_tensor)
        #     pred = torch.argmax(logits, dim=1).item()
        #     gesture = idx_to_class[str(pred)]

        frame = cv2.flip(frame, 1)
        # Display the gesture on the video stream
        # cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Detect hand landmarks from the input image.
        # Convert the frame to RGB and wrap it in a Mediapipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detection_result = detector.detect(mp_image)
        height, width, _ = frame.shape

        if len(detection_result.hand_landmarks) > 0:
            
            hand_landmarks = detection_result.hand_landmarks[0]
            xs = [lm.x * width  for lm in hand_landmarks]
            ys = [lm.y * height for lm in hand_landmarks]

            left_x, right_x = min(xs), max(xs)
            top_y, bottom_y = min(ys), max(ys)

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

            # Draw rectangle over the hand
            cv2.rectangle(frame, (left_x, top_y), (right_x, bottom_y), (255, 0, 0), 2)

        cv2.imshow('Live Video Stream', frame)

        # Quit if q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()