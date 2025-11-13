import cv2 

FRAME_WIDTH = 200
FRAME_HEIGHT = 300


capture = cv2.VideoCapture(0)

while (True):
    # Store the frame from the video capture and resize it to the desired window size.
    ret, frame = capture.read()
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    cv2.imshow("Camera Input", frame)
    # Check if user wants to exit.
    if (cv2.waitKey(1) & 0xFF == ord('x')):
        break

# When we exit the loop, we have to stop the capture too.
capture.release()
cv2.destroyAllWindows()