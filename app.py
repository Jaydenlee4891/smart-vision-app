from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

import cv2
import numpy as np
import time
from plyer import tts
import tensorflow as tf


# -----------------------------
# YOLO Class Names
# -----------------------------
NAMES = {
    0: "Bike Front",
    1: "Bike Left",
    2: "Bike Right",
    3: "Car Front",
    4: "Car Left",
    5: "Car Right",
    6: "Crossroad",
    7: "Fence Front",
    8: "Fence Left",
    9: "Fence Right",
    10: "Pedestrian Light Green",
    11: "Pedestrian Light Red",
    12: "Person Front",
    13: "Person Left",
    14: "Person Right",
    15: "Pole Front",
    16: "Pole Left",
    17: "Pole Right",
    18: "Traffic Cone Right",
    19: "Traffic Light Green",
    20: "Traffic Light Orange",
    21: "Traffic Light Red",
    22: "Trash Left",
    23: "Trash Right",
    24: "Tree Front",
    25: "Tree Right"
}

# -----------------------------
# Globals
# -----------------------------
running = False
cap = None
last_spoken = ""
speak_delay = 2
last_speak_time = 0


# -----------------------------
# Load TFLite model
# -----------------------------
interpreter = tf.lite.Interpreter(
    model_path="/Users/apple/Downloads/kivytry/best_float32.tflite"
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = input_details[0]['shape'][1]  # should be 640


# -----------------------------
# Functions
# -----------------------------
def speak(text):
    """Speak labels with delay to avoid repetition."""
    global last_spoken, last_speak_time
    if text != last_spoken and (time.time() - last_speak_time) > speak_delay:
        last_spoken = text
        last_speak_time = time.time()
        try:
            tts.speak(text)
        except:
            pass


def process_yolo_output():
    """
    Decodes TFLite YOLO output of shape [1, 30, 8400].
    Format per prediction:
        0-3: x_center, y_center, w, h
        4-29: class scores (26 classes)
    """
    output = interpreter.get_tensor(output_details[0]['index'])[0]  # shape: [30, 8400]

    # transpose → [8400, 30]
    output = output.T

    boxes = output[:, 0:4]
    class_probs = output[:, 4:]

    classes = np.argmax(class_probs, axis=-1)
    scores = np.max(class_probs, axis=-1)

    return boxes, scores, classes


# -----------------------------
# Kivy App
# -----------------------------
class CameraApp(App):

    def build(self):
        self.layout = BoxLayout(orientation='vertical')

        # Image widget for camera
        self.img_widget = Image()
        self.layout.add_widget(self.img_widget)

        # Buttons
        btn_layout = BoxLayout(size_hint_y=0.15)
        start_btn = Button(text='Start')
        start_btn.bind(on_press=self.start_camera)
        stop_btn = Button(text='Stop')
        stop_btn.bind(on_press=self.stop_camera)
        btn_layout.add_widget(start_btn)
        btn_layout.add_widget(stop_btn)
        self.layout.add_widget(btn_layout)

        Clock.schedule_interval(self.update, 1.0 / 30.0)

        return self.layout

    def start_camera(self, instance):
        global running, cap
        if running:
            return
        running = True
        cap = cv2.VideoCapture(0)

    def stop_camera(self, instance):
        global running, cap
        running = False
        if cap:
            cap.release()
            cap = None

    def update(self, dt):
        global running, cap
        if running and cap:
            ret, frame = cap.read()
            if ret:
            
                frame = cv2.flip(frame, 0)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Preprocess for YOLO
                resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
                input_data = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0

                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                # Decode detections
                boxes, scores, classes = process_yolo_output()

                # Speak detected labels
                for box, score, cls in zip(boxes, scores, classes):
                    if score < 0.25:   
                        continue

                    label = NAMES.get(int(cls), "Unknown")
                    speak(label)
                    break  # speak only ONE per frame

                # Output camera feed to UI
                buf = rgb.flatten()
                texture = Texture.create(size=(rgb.shape[1], rgb.shape[0]), colorfmt='rgb')
                texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
                self.img_widget.texture = texture


# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    CameraApp().run()

