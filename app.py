from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

import cv2
import numpy as np
import time
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

CONF_THRESHOLD = 0.30
NMS_THRESHOLD = 0.45


# -----------------------------
# Load TFLite model
# -----------------------------
interpreter = tf.lite.Interpreter(
    model_path="/Users/apple/Downloads/kivytry/best_float32.tflite"
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = input_details[0]['shape'][1]  # usually 640


# -----------------------------
# YOLO Decode + NMS
# -----------------------------
def process_yolo_output():
    """
    Output shape: (1, 30, 8400)
    Format: x, y, w, h, obj, class_probs...
    """
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    output = output.T  # (8400, 30)

    boxes = output[:, 0:4]
    objectness = output[:, 4]
    class_probs = output[:, 5:]

    class_ids = np.argmax(class_probs, axis=1)
    class_scores = np.max(class_probs, axis=1)
    scores = objectness * class_scores

    return boxes, scores, class_ids


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [
        float(x - w / 2),
        float(y - h / 2),
        float(w),
        float(h),
    ]


# -----------------------------
# Kivy App
# -----------------------------
class CameraApp(App):

    def build(self):
        self.layout = BoxLayout(orientation='vertical')

        self.img_widget = Image()
        self.layout.add_widget(self.img_widget)

        btn_layout = BoxLayout(size_hint_y=0.15)
        start_btn = Button(text='Start')
        stop_btn = Button(text='Stop')

        start_btn.bind(on_press=self.start_camera)
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

        if not (running and cap):
            return

        ret, frame = cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 0)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        input_data = np.expand_dims(resized, axis=0).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes, scores, class_ids = process_yolo_output()

        # Prepare boxes for NMS
        nms_boxes = []
        nms_scores = []

        for box, score in zip(boxes, scores):
            if score < CONF_THRESHOLD:
                continue
            nms_boxes.append(xywh_to_xyxy(box))
            nms_scores.append(float(score))

        indices = cv2.dnn.NMSBoxes(
            bboxes=nms_boxes,
            scores=nms_scores,
            score_threshold=CONF_THRESHOLD,
            nms_threshold=NMS_THRESHOLD
        )

        if len(indices) > 0:
            for i in indices.flatten():
                cls = class_ids[i]
                label = NAMES.get(int(cls), "Unknown")
                conf = nms_scores[i]
                print(f"[DETECTION] {label} | {conf:.2f}")

        # Display camera feed
        buf = rgb.flatten()
        texture = Texture.create(size=(rgb.shape[1], rgb.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.img_widget.texture = texture


# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    CameraApp().run()
