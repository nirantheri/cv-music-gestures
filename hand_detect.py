import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import threading
import pyaudio
from pydub import AudioSegment
from pydub.generators import Sine
from moviepy import VideoFileClip, AudioFileClip
import warnings
warnings.filterwarnings("ignore")

COLOR = (138, 33, 224)
FANCY_FONT = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
READABLE_FONT = cv2.FONT_HERSHEY_DUPLEX

GESTURES = {"None": "no gesture detected", 
            "Closed_Fist":"fist", 
            "Open_Palm": "five", 
            "Pointing_Up":"one", 
            "Thumb_Down": "dislike", 
            "Thumb_Up": "like", 
            "Victory": "two", 
            "ILoveYou": "ily"}

MODEL_PATH = r'gesture_recognizer.task'

audio_frames = []
audio_segment = AudioSegment.silent(duration=0)
recording_flag = False
audio_stop_flag = False
current_gesture = "None"

CHUNK = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

def synth_sound(frame_segment):
    sine = Sine(440).to_audio_segment(duration=len(frame_segment)).apply_gain(-15) # meant to smooth the sine
    frame_segment = frame_segment.overlay(sine)

    return frame_segment
def apply_choral_effect(segment):
    # speed change to shift pitch (slightly up and down)
    def pitch_shift(seg, semitones):
        factor = 2 ** (semitones / 12)
        new_rate = round(seg.frame_rate * factor)
        shifted = seg._spawn(seg.raw_data, overrides={'frame_rate': new_rate})
        return shifted.set_frame_rate(seg.frame_rate)
    
    segment = segment.fade_in(3).fade_out(3)
    maj3 = pitch_shift(segment, 4)
    p5 = pitch_shift(segment, 7)

    # semitones to octave conversion

    return segment.overlay(maj3).overlay(p5)
def record_audio():
    global audio_segment, audio_stop_flag, current_gesture

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)

    while not audio_stop_flag:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frame_segment = AudioSegment(
            data,
            sample_width=p.get_sample_size(FORMAT),
            frame_rate=RATE,
            channels=CHANNELS
        )

        if current_gesture == "Pointing_Up":
            frame_segment = apply_choral_effect(frame_segment)
        elif current_gesture == "Victory":
            frame_segment = synth_sound(frame_segment)

        audio_segment += frame_segment

    stream.stop_stream()
    stream.close()
    p.terminate()

def init_recognizer():


    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    GestureRecognizer = vision.GestureRecognizer
    GestureRecognizerOptions = vision.GestureRecognizerOptions
    VisionRunningMode = vision.RunningMode

    options = GestureRecognizerOptions(
        base_options=base_options,
        running_mode=VisionRunningMode.IMAGE,
    )

    recognizer = GestureRecognizer.create_from_options(options)
    print("intialized model")
    return recognizer

def main(recognizer):
    global recording_flag, audio_stop_flag, current_gesture, audio_segment
    cap = cv2.VideoCapture(0)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    audio_thread = None

    print("Press 'q' to exit the video stream.")

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
            
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        detection_result = recognizer.recognize(mp_image)

        height, width, _ = frame.shape

        if (detection_result.gestures):
            gesture_label = GESTURES[detection_result.gestures[0][0].category_name]
            current_gesture = detection_result.gestures[0][0].category_name
            
            hand_landmark = detection_result.hand_landmarks[0]

            xs = [lm.x * width  for lm in hand_landmark]
            ys = [lm.y * height for lm in hand_landmark]

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


            cv2.rectangle(frame, (left_x, top_y), (right_x, bottom_y), COLOR, 2)
            cv2.putText(frame, gesture_label, (right_x, top_y-10),
                        READABLE_FONT, 0.8, COLOR, 2)
            
        # Start Recording
        if current_gesture == "Open_Palm" and not recording_flag:
            print("Recording started!")
            out = cv2.VideoWriter('gesture_output.mp4', fourcc, 20.0, (frame_width, frame_height))
            audio_frames.clear()
            audio_segment = AudioSegment.silent(duration=0)
            audio_stop_flag = False
            audio_thread = threading.Thread(target=record_audio)
            audio_thread.start()
            recording_flag = True
        
        # Stop Recording

        if current_gesture == "Closed_Fist" and recording_flag:
            print("Recording stopped!")
            recording_flag = False
            if out: out.release()
            audio_stop_flag = True
            audio_thread.join()
            # Save audio
            audio_segment.export("gesture_audio.wav", format="wav")
            # Merge video + audio
            video_clip = VideoFileClip('gesture_output.mp4')
            audio_clip = AudioFileClip('gesture_audio.wav')
            final_clip = video_clip.with_audio(audio_clip)
            final_clip.write_videofile('gesture_final.mp4', codec='libx264', audio_codec='aac')
            print("Saved gesture_final.mp4")
            break
        
        if recording_flag and out:
            cv2.putText(frame, "Recording...", (10, 30), FANCY_FONT, 1, COLOR, 2)
            out.write(frame)


        cv2.imshow('Live Video Stream', frame)

        # Quit if q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if recording_flag:
                print("Recording stopped!")
                recording_flag = False
                if out: out.release()
                audio_stop_flag = True
                audio_thread.join()
                # Save audio
                audio_segment.export("gesture_audio.wav", format="wav")
                # Merge video + audio
                video_clip = VideoFileClip('gesture_output.mp4')
                audio_clip = AudioFileClip('gesture_audio.wav')
                final_clip = video_clip.with_audio(audio_clip)
                final_clip.write_videofile('gesture_final.mp4', codec='libx264', audio_codec='aac')
                print("Saved gesture_final.mp4")
            break
        
    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()
    if out: 
        out.release()

if __name__ == "__main__":
    recog_model = init_recognizer()
    main(recog_model)